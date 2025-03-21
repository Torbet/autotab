import torch
import torch.nn.functional as F
import numpy as np


def transcribe_beam_search(
  model, tokenizer, audio: torch.Tensor, beam_size=5, max_length=1000, temp=0.5, no_repeat_ngram_size=3, repetition_penalty=0.9
):
  model.eval()
  device = audio.device
  encoded_audio = model.encoder(audio.unsqueeze(0))
  start_token = tokenizer._special_tokens['<|startoftab|>']
  end_token = tokenizer._special_tokens['<|endoftab|>']

  beams = [([start_token], 0.0)]

  for _ in range(max_length):
    new_beams = []
    for tokens, score in beams:
      if tokens[-1] == end_token:
        new_beams.append((tokens, score))
        continue

      tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
      logits = model.decoder(tokens_tensor, encoded_audio)
      logits = logits[0, -1, :] / temp  # Apply temperature scaling

      for token in set(tokens):
        logits[token] = logits[token] / repetition_penalty

      probs = F.softmax(logits, dim=-1)

      top_probs, top_indices = torch.topk(probs, beam_size)
      for i in range(beam_size):
        new_token = top_indices[i].item()

        if len(tokens) >= no_repeat_ngram_size - 1:
          ngram = tokens[-(no_repeat_ngram_size - 1) :] + [new_token]
          repeat = False
          for j in range(len(tokens) - no_repeat_ngram_size + 1):
            if tokens[j : j + no_repeat_ngram_size] == ngram:
              repeat = True
              break
          if repeat:
            continue

        new_score = (score + torch.log(top_probs[i]).item()) / (len(tokens) + 1) ** 0.5
        new_beams.append((tokens + [new_token], new_score))

    if not new_beams:
      break

    beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

    if all(tokens[-1] == end_token for tokens, _ in beams):
      break

  best_tokens, best_score = beams[0]
  transcription = tokenizer.decode(np.array(best_tokens))
  return transcription, best_tokens


def transcribe_greedy(model, tokenizer, audio: torch.Tensor, max_length=1000):
  model.eval()
  tokens = torch.tensor([tokenizer._special_tokens['<|startoftab|>']]).unsqueeze(0).to(audio.device)
  encoded_audio = model.encoder(audio.unsqueeze(0))

  for _ in range(max_length - 1):
    logits = model.decoder(tokens, encoded_audio)
    logits = logits[0, -1, :]
    token = logits.argmax().item()
    tokens = torch.cat([tokens, torch.tensor([token]).unsqueeze(0).to(tokens.device)], dim=1)
    if token == tokenizer._special_tokens['<|endoftab|>']:
      break

  transcription = tokenizer.decode(tokens[0].cpu().numpy())
  return transcription, tokens[0].cpu().numpy()


def transcribe_multinomial(model, tokenizer, audio: torch.Tensor, max_length=1000, temp=0.5):
  model.eval()
  tokens = torch.tensor([tokenizer._special_tokens['<|startoftab|>']]).unsqueeze(0).to(audio.device)
  encoded_audio = model.encoder(audio.unsqueeze(0))

  for _ in range(max_length - 1):
    logits = model.decoder(tokens, encoded_audio)
    logits = logits[0, -1, :] / temp
    token = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
    tokens = torch.cat([tokens, torch.tensor([token]).unsqueeze(0).to(tokens.device)], dim=1)
    if token == tokenizer._special_tokens['<|endoftab|>']:
      break

  transcription = tokenizer.decode(tokens[0].cpu().numpy())
  return transcription, tokens[0].cpu().numpy()


if __name__ == '__main__':
  import numpy as np
  from train import get_model

  scheduled_sampling = False
  fine_tune = True
  path = 'results/tiny.pth'
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  weights = torch.load(path, map_location=device)
  weights = {k.replace('module.', ''): v for k, v in weights.items()}
  model, tokenizer = get_model('tiny')
  model.load_state_dict(weights)
  model.to(device)

  data = np.load('data/raw/audio_tabs_batch_1.npz')
  # data = np.load('data/test_data_solo.npz')
  idx = 0
  tabs = data['tabs']
  audios = torch.from_numpy(data['audio']).float()
  tab = tabs[idx]
  tab = tab[tab != 0]
  audio = audios[0].to(device)

  print('Gold Tab', tokenizer.decode(tab), '\n\n')

  transcription, tokens = transcribe_beam_search(
    model, tokenizer, audio, beam_size=5, max_length=1000, temp=0.5, no_repeat_ngram_size=3, repetition_penalty=1.0
  )
  print('Transcription (beam search):', transcription, '\n\n')

  transcription, tokens = transcribe_greedy(model, tokenizer, audio, max_length=1000)
  print('Transcription (greedy):', transcription, '\n\n')

  transcription, tokens = transcribe_multinomial(model, tokenizer, audio, max_length=1000, temp=0.5)
  print('Transcription (multinomial):', transcription, '\n\n')
