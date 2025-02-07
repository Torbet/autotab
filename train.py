import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as DS, DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
from model import Model

# config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 20
lr = 1e-4
batch_size = 64


class Dataset(DS):
  def __init__(self):
    data = np.load('data.npz')
    self.waves = torch.tensor(data['waves'], dtype=torch.float32)
    self.tabs = torch.tensor(data['tabs'], dtype=torch.float32)

  def __len__(self):
    return len(self.waves)

  def __getitem__(self, idx):
    return self.waves[idx], self.tabs[idx]


def split(dataset: Dataset, batch_size: int = 64):
  n = len(dataset)
  t = int(n * 0.8)
  train, val, test = random_split(dataset, [t, n - t, n - t])
  return (DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in (train, val, test))


def train(model: Model, loader: DataLoader, optimizer: optim.Optimizer) -> tuple[float, float]:
  # TODO: no grad encoder and decoder separately
  model.train()
  for wave, tab in (t := tqdm(loader)):
    wave, tab = wave.to(device), tab.to(device)
    optimizer.zero_grad()
    output = model(wave)
    loss = F.mse_loss(output, tab)
    loss.backward()
    optimizer.step()
    t.set_description(f'Loss: {loss.item():.2f}')
  return evaluate(model, loader)


def evaluate(model: Model, loader: DataLoader) -> tuple[float, float]:
  loss, correct, total = 0, 0, 0
  model.eval()
  with torch.no_grad():
    for wave, tab in loader:
      wave, tab = wave.to(device), tab.to(device)
      output = model(wave)
      loss += F.mse_loss(output, tab, reduction='sum').item()
      correct += (output.argmax(1) == tab).sum().item()
      total += tab.size(0)
  return loss / total, correct / total


if __name__ == '__main__':
  print(f'device: {device}, epochs: {epochs}, lr: {lr}, batch_size: {batch_size}')
  model = Model().to(device)
  dataset = Dataset()
  train_loader, val_loader, test_loader = split(dataset, batch_size)
  optimizer = optim.AdamW(model.parameters(), lr=lr)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

  for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader, optimizer)
    val_loss, val_acc = evaluate(model, val_loader)
    print(f'Epoch: {epoch + 1} | Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f} | Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f}')
    scheduler.step(val_loss)
  test_loss, test_acc = evaluate(model, test_loader)
  print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc:.2f}')
  torch.save(model.state_dict(), 'model.pth')
