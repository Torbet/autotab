from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from tqdm import tqdm
from model import Transformer
import matplotlib.pyplot as plt
import os
from tokenizer import encoder

torch.cuda.empty_cache()
print(f'cuda memory allocated: {torch.cuda.memory_allocated()}')
print(f'cuda memory cached: {torch.cuda.memory_reserved()}')

np.random.seed(0)
torch.manual_seed(0)

teacher_forcing = True
fine_tune = False

t = encoder()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 20
lr = 1e-4
weight_decay = 1e-3
grad_clip = 1.0
batch_size = 16
n_vocab = t.n_vocab
text_ctx = 1000
dims = {  # tiny.en whisper with tweaked context sizes
  'n_mels': 80,
  'n_vocab': n_vocab,
  'n_audio_ctx': 1500,
  'n_audio_state': 384,
  'n_audio_head': 6,
  'n_audio_layer': 4,
  'n_text_ctx': text_ctx,
  'n_text_state': 384,
  'n_text_head': 6,
  'n_text_layer': 4,
}
start_token = t._special_tokens['<|startoftab|>']


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

    self.idx_map = []
    for i, (tabs, _) in enumerate(self.data):
      for j in range(tabs.shape[0]):
        self.idx_map.append((i, j))

  def __len__(self):
    return len(self.idx_map)

  def __getitem__(self, idx):
    i, j = self.idx_map[idx]
    tabs, audio = self.data[i]
    return torch.from_numpy(tabs[j]).long(), torch.from_numpy(audio[j]).float()


def split(dataset: data.Dataset):
  l = len(dataset)
  train_size = int(l * 0.8)
  val_size = int(l * 0.1)
  test_size = l - train_size - val_size
  train, val, test = data.random_split(dataset, [train_size, val_size, test_size])
  return (data.DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in (train, val, test))


def train(model: Transformer, loader: data.DataLoader, optimizer: optim.Optimizer, rate: float) -> tuple[float, float]:
  model.train()
  for tab, audio in (t := tqdm(loader)):
    tab, audio = tab.to(device), audio.to(device)
    optimizer.zero_grad()
    logits = model(audio, tab[:, :-1], teacher_forcing, rate)
    loss = F.cross_entropy(logits.view(-1, n_vocab), tab[:, 1:].flatten(), ignore_index=0)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    t.set_description(f'loss: {loss.item():.4f}')
  return evaluate(model, loader)


def evaluate(model: Transformer, loader: data.DataLoader) -> tuple[float, float]:
  model.eval()
  loss, correct, total = 0, 0, 0
  with torch.no_grad():
    for tab, audio in tqdm(loader):
      tab, audio = tab.to(device), audio.to(device)
      logits = model(audio, tab[:, :-1], teacher_forcing=False)
      loss += F.cross_entropy(logits.view(-1, n_vocab), tab[:, 1:].flatten(), ignore_index=0).item()
      correct += (logits.argmax(-1) == tab[:, 1:]).sum().item()
      total += tab[:, 1:].numel()
  return loss / len(loader), correct / total


if __name__ == '__main__':
  print(f'device: {device}, epochs: {epochs}, lr: {lr}, batch_size: {batch_size}, grad_clip: {grad_clip}')
  print(f'dims: {dims}')
  model = nn.DataParallel(Transformer(dims)).to(device)
  optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-9)
  scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=100, epochs=epochs)
  dataset = Dataset()
  train_loader, val_loader, test_loader = split(dataset)

  results = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
  }

  for epoch in range(epochs):
    rate = 1.0 - (epoch / epochs)
    print(f'Epoch {epoch + 1}/{epochs} -- Teacher Forcing Rate: {rate:.2f}')
    train_loss, train_acc = train(model, train_loader, optimizer, rate)
    scheduler.step()
    val_loss, val_acc = evaluate(model, val_loader)
    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['val_loss'].append(val_loss)
    results['val_acc'].append(val_acc)
    print(f'train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}\n')

  test_loss, test_acc = evaluate(model, test_loader)
  print(f'test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}')

  name = f'model{"_tf" if teacher_forcing else ""}{"_ft" if fine_tune else ""}'

  torch.save(model.state_dict(), f'{name}.pth')

  plt.plot(results['train_loss'], label='train_loss')
  plt.plot(results['val_loss'], label='val_loss')
  plt.plot(results['train_acc'], label='train_acc')
  plt.plot(results['val_acc'], label='val_acc')
  plt.legend()
  plt.savefig(f'{name}.png')
