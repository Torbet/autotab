import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn import metrics
import editdistance
from model import Transformer
from train import evaluate
import csv

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

models = ['tiny', 'tiny_ft', 'tiny_ss', 'tiny_ss_ft', 'tiny_ss_ft_freeze', 'tiny_ft_freeze']
results = {}

for name in models:
  print(f'Evaluating {name}')
  model = Transformer(dims)
  model.load_state_dict(torch.load(f'results/{name}.pth'))
  model.to(device)
  results[name] = evaluate(model, loader)

with open('results.csv', 'w') as f:
  writer = csv.writer(f)
  keys = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'wer']
  writer.writerow(['model'] + keys)
  for name, result in results.items():
    writer.writerow([name] + [result[k] for k in keys])
