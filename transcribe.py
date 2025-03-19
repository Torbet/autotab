import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Transformer
from tokenizer import encoder
from train import get_model
import tiktoken

scheduled_sampling = False
fine_tune = False
path = f'results/tiny{"_ss" if scheduled_sampling else ""}{"_ft" if fine_tune else ""}_20_0.0001_0.001_1.0_16.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = torch.load(path, map_location=device, weights_only=True)
weights = {k.replace('module.', ''): v for k, v in weights.items()}
model, tokenizer = get_model('tiny')
model.load_state_dict(weights)
model.to(device)


def transcribe(model: Transformer, tokenizer: tiktoken.Encoding, audio):
  temperature = 0.5
  encoded_audio = model.encoder(audio.unsqueeze(0))
  tokens = torch.tensor([tokenizer._special_tokens['<|startoftab|>']]).to(device)
  for _ in range(1000):
    logits = model.decoder(tokens.unsqueeze(0), encoded_audio)
    logits = logits[0, -1, :] / temperature
    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    # token = torch.argmax(probs).unsqueeze(0)
    tokens = torch.cat([tokens, token])
    if token == tokenizer._special_tokens['<|endoftab|>']:
      break
  return tokenizer.decode(tokens.cpu().numpy()), tokens


if __name__ == '__main__':
  tokenizer = encoder()
  data = np.load('data/raw/audio_tabs_batch_10.npz')
  tabs = torch.from_numpy(data['tabs']).long()
  audios = torch.from_numpy(data['audio']).float()
  idx = 0
  tab = tabs[idx]
  tab = tab[tab != 0].cpu().numpy()
  audio = audios[idx].to(device)

  print(tokenizer.decode(tab))
  print()

  text, tokens = transcribe(model, tokenizer, audio)
  print(text)

  print(tab.shape, tokens.shape)

  # make tokens and tab same length
  tab = tab[: len(tokens)]
  tokens = tokens.cpu().numpy()
  tokens = tokens[: len(tab)]

  accuracy = (tab == tokens).sum() / len(tab)
  print(f'Accuracy: {accuracy:.2f}')
