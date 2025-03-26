import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn import metrics
import editdistance
from model import Transformer
from train import evaluate
import csv

np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dims = {
  'n_mels': 80,
  'n_vocab': 137,
  'n_audio_ctx': 1500,
  'n_audio_state': 384,
  'n_audio_head': 6,
  'n_audio_layer': 4,
  'n_text_ctx': 1000,
  'n_text_state': 384,
  'n_text_head': 6,
  'n_text_layer': 4,
}


class Dataset(data.Dataset):
  def __init__(self):
    files = ['data/test_data_solo.npz', 'data/test_data_comp.npz']
    self.tabs = []
    self.audio = []
    for f in files:
      d = np.load(f)
      self.tabs.extend(d['tabs'])
      self.audio.extend(d['audio'])

  def __len__(self):
    return len(self.tabs)

  def __getitem__(self, idx):
    return torch.from_numpy(self.tabs[idx]).long(), torch.from_numpy(self.audio[idx]).float()


loader = data.DataLoader(Dataset(), batch_size=8, shuffle=True)

sizes = [round(0.2 * i, 1) for i in range(1, 6)]
results = {}

for size in sizes:
  print(f'Evaluating {size}')
  model = Transformer(dims)
  model.load_state_dict(torch.load(f'results/tiny_{size}.pth'))
  model.to(device)
  results[size] = evaluate(model, loader)

with open('results.csv', 'w') as f:
  writer = csv.writer(f)
  keys = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'wer']
  writer.writerow(['model'] + keys)
  for name, result in results.items():
    writer.writerow([name] + [result[k] for k in keys])

# log confusion matrix
for name, result in results.items():
  with open(f'{name}_confusion.txt', 'w', newline='') as f:
    f.write(str(result['confusion']))
