from __future__ import annotations
import argparse
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import Transformer
from tokenizer import encoder
from sklearn import metrics
import editdistance

# Clear CUDA cache and print memory status.
torch.cuda.empty_cache()
print(f'cuda memory allocated: {torch.cuda.memory_allocated()}')
print(f'cuda memory cached: {torch.cuda.memory_reserved()}')

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
# boolean optional, default False
parser.add_argument('--ss', action='store_true', help='Use scheduled sampling')
parser.add_argument('--ft', action='store_true', help='Fine tune the encoder')
parser.add_argument('--freeze', action='store_true', help='Freeze the encoder')
parser.add_argument('--model', type=str, default='tiny', help='Model name')
args = parser.parse_args()

scheduled_sampling = args.ss
fine_tune = args.ft
freeze = args.freeze
model_name = args.model
epochs = 20
lr = 1e-4
weight_decay = 1e-3
batch_size = 8
grad_clip = 1.0
n_text_ctx = 1000

MODEL_URLS = {
  'tiny.en': 'https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt',
  'tiny': 'https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt',
  'base.en': 'https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt',
  'base': 'https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt',
  'small.en': 'https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt',
  'small': 'https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt',
  'medium.en': 'https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt',
  'medium': 'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt',
  'large-v1': 'https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt',
  'large-v2': 'https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt',
  'large': 'https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt',
}


def get_model(name):
  tokenizer = encoder()
  state = torch.hub.load_state_dict_from_url(MODEL_URLS[name])
  state['dims']['n_vocab'] = tokenizer.n_vocab
  state['dims']['n_text_ctx'] = n_text_ctx
  model = Transformer(state['dims'])
  if fine_tune:
    model.encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in state['model_state_dict'].items() if 'encoder' in k})
    if freeze:
      for param in model.encoder.parameters():
        param.requires_grad = False
  return model, tokenizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, tokenizer = get_model(model_name)
start_token = tokenizer._special_tokens['<|startoftab|>']


class Dataset(data.Dataset):
  def __init__(self):
    self.paths = sorted([os.path.join('data/raw', f) for f in os.listdir('data/raw') if f.endswith('.npz')])
    self.data = []
    for path in self.paths:
      try:
        print(f'Loading {path}')
        d = np.load(path)
        self.data.append((d['tabs'], d['audio']))
      except Exception as e:
        print(f'Error loading {path}: {e}')
    self.idx_map = [(i, j) for i, (tabs, _) in enumerate(self.data) for j in range(tabs.shape[0])]

  def __len__(self):
    return len(self.idx_map)

  def __getitem__(self, idx):
    i, j = self.idx_map[idx]
    tabs, audio = self.data[i]
    return torch.from_numpy(tabs[j]).long(), torch.from_numpy(audio[j]).float()


def split(dataset: data.Dataset):
  l = len(dataset)
  sizes = [int(l * 0.8), int(l * 0.1), l - int(l * 0.8) - int(l * 0.1)]
  train, val, test = data.random_split(dataset, sizes)
  return (data.DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in (train, val, test))


def train(model: Transformer, loader: data.DataLoader, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler, rate: float) -> dict:
  model.train()
  for tab, audio in tqdm(loader, desc='Training'):
    tab, audio = tab.to(device), audio.to(device)
    optimizer.zero_grad()
    logits = model(audio, tab[:, :-1])

    if scheduled_sampling:
      mask = torch.rand_like(tab[:, 1:].float()) < rate
      teacher = tab[:, 1:]
      student = logits.argmax(-1)
      mixed = torch.cat([tab[:, :1], torch.where(mask, teacher, student)], dim=1)
      logits = model(audio, mixed[:, :-1])

    loss = F.cross_entropy(logits.view(-1, tokenizer.n_vocab), tab[:, 1:].flatten(), ignore_index=0)
    loss.backward()
    optimizer.step()
    scheduler.step()
  return evaluate(model, loader)


def evaluate(model: Transformer, loader: data.DataLoader) -> dict:
  model.eval()
  loss, correct, total = 0.0, 0, 0
  labels, preds = [], []
  wer = []

  with torch.no_grad():
    for tab, audio in tqdm(loader, desc='Evaluating'):
      tab, audio = tab.to(device), audio.to(device)
      logits = model(audio, tab[:, :-1])
      predictions = logits.argmax(-1)
      loss += F.cross_entropy(logits.view(-1, tokenizer.n_vocab), tab[:, 1:].flatten(), ignore_index=0).item()
      mask = tab[:, 1:] != 0
      correct += (predictions == tab[:, 1:]).float().masked_select(mask).sum().item()
      total += mask.sum().item()
      labels.extend(tab[:, 1:].flatten().cpu().tolist())
      preds.extend(predictions.flatten().cpu().tolist())

      for i in range(tab.size(0)):
        ref_tokens = [t for t in tab[i, 1:].cpu().tolist() if t != 0]
        pred_tokens = [t for t in predictions[i].cpu().tolist() if t != 0]
        wer.append(editdistance.eval(ref_tokens, pred_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0)

  return {
    'loss': loss / len(loader),
    'accuracy': correct / total,
    'precision': metrics.precision_score(labels, preds, average='weighted', zero_division=0),
    'recall': metrics.recall_score(labels, preds, average='weighted', zero_division=0),
    'f1': metrics.f1_score(labels, preds, average='weighted', zero_division=0),
    'wer': np.mean(wer),
    'confusion': metrics.confusion_matrix(labels, preds).tolist(),
  }


if __name__ == '__main__':
  print(f'Model: {model_name}, Scheduled Sampling: {scheduled_sampling}, Fine Tuning: {fine_tune}, Freeze: {freeze}')
  print(f'Epochs: {epochs}, LR: {lr}, Weight Decay: {weight_decay}, Grad Clip: {grad_clip}, Batch Size: {batch_size}')
  print(f'Device: {device}')

  dataset = Dataset()
  train_loader, val_loader, test_loader = split(dataset)
  model = nn.DataParallel(model).to(device)
  optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-9)
  scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(train_loader))

  results = {}
  for epoch in range(epochs):
    rate = 1.0 - (epoch / epochs)
    # rate = 1.0 if epoch < 5 else 1.0 - (epoch - 5) / (epochs - 5)

    print(f'\nEpoch {epoch + 1}/{epochs} -- Teacher Forcing Rate: {rate:.2f}')
    train_results = train(model, train_loader, optimizer, scheduler, rate)
    val_results = evaluate(model, val_loader)
    results[epoch] = {'train': train_results, 'val': val_results}
    print(f'Train: Loss {train_results["loss"]:.4f}, Acc {train_results["accuracy"]:.4f}, WER {train_results["wer"]:.4f}')
    print(f'Val: Loss {val_results["loss"]:.4f}, Acc {val_results["accuracy"]:.4f}, WER {val_results["wer"]:.4f}')
  test_results = evaluate(model, test_loader)
  print(f'Test: Loss {test_results["loss"]:.4f}, Acc {test_results["accuracy"]:.4f}, WER {test_results["wer"]:.4f}')

  name = f'results/{model_name}{"_ss" if scheduled_sampling else ""}{"_ft" if fine_tune else ""}{f"_freeze" if freeze else ""}'
  torch.save(model.module.state_dict(), f'{name}.pth')

  # Save CSV results.
  keys = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'wer']
  with open(f'{name}.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'type'] + keys)
    for epoch in range(epochs):
      for split_type in ('train', 'val'):
        writer.writerow([epoch, split_type] + [results[epoch][split_type][k] for k in keys])
    writer.writerow(['+', 'test'] + [test_results[k] for k in keys])

  # Save test confusion matrix.
  with open(f'{name}_confusion.txt', 'w', newline='') as f:
    f.write(str(test_results['confusion']))

  # Plot training curves.
  plt.figure()
  for key in keys:
    plt.plot([results[e]['train'][key] for e in range(epochs)], label=f'train_{key}')
    plt.plot([results[e]['val'][key] for e in range(epochs)], label=f'val_{key}')
  plt.legend()
  plt.savefig(f'{name}.png')
