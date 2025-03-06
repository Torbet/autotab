import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import librosa
from model import Transformer
from tokenizer import encoder
from train import dims
import tiktoken

SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
SAMPLES_PER_SEGMENT = SAMPLE_RATE * 30
FRAMES_PER_SEGMENT = SAMPLES_PER_SEGMENT // HOP_LENGTH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = torch.load('model.pth', map_location=device, weights_only=True)
model = nn.DataParallel(Transformer(dims)).to(device)
model.load_state_dict(weights)


def transcribe(model: nn.DataParallel, tokenizer: tiktoken.Encoding, audio):
  encoded_audio = model.module.encoder(audio.unsqueeze(0))
  tokens = torch.tensor([tokenizer._special_tokens['<|startoftab|>']]).to(device)
  max_tokens = dims['n_text_ctx']
  for _ in range(max_tokens):
    logits = model.module.decoder(tokens.unsqueeze(0), encoded_audio)
    logits = logits[0, -1]
    token = torch.argmax(logits)
    tokens = torch.cat([tokens, token.unsqueeze(0)])
    if token == tokenizer._special_tokens['<|endoftab|>']:
      break
  return tokenizer.decode(tokens.cpu().numpy()), tokens


if __name__ == '__main__':
  tokenizer = encoder()
  data = np.load('data/raw/audio_tabs_batch_10.npz')
  tabs = torch.from_numpy(data['tabs']).long()
  audios = torch.from_numpy(data['audio']).float()
  idx = 2
  tab = tabs[idx]
  tab = tab[tab != 0].cpu().numpy()
  audio = audios[idx].to(device)

  print(tokenizer.decode(tab))
  print()

  text, tokens = transcribe(model, tokenizer, audio)
  print(text)

  # make tokens and tab same length
  tokens = tokens[1:]
  tokens = tokens[: len(tab)]

  accuracy = (tab == tokens.cpu().numpy()).sum() / len(tab)
  print(f'Accuracy: {accuracy:.2f}')
