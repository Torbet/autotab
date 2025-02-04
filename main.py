import torch
import torch.nn.functional as F
import tiktoken
import requests
import base64
import itertools
from model import Model
from audio import load_audio
from tokenizer import encoder

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

LANGUAGES = {
  'en': 'english',
  'zh': 'chinese',
  'de': 'german',
  'es': 'spanish',
  'ru': 'russian',
  'ko': 'korean',
  'fr': 'french',
  'ja': 'japanese',
  'pt': 'portuguese',
  'tr': 'turkish',
  'pl': 'polish',
  'ca': 'catalan',
  'nl': 'dutch',
  'ar': 'arabic',
  'sv': 'swedish',
  'it': 'italian',
  'id': 'indonesian',
  'hi': 'hindi',
  'fi': 'finnish',
  'vi': 'vietnamese',
  'he': 'hebrew',
  'uk': 'ukrainian',
  'el': 'greek',
  'ms': 'malay',
  'cs': 'czech',
  'ro': 'romanian',
  'da': 'danish',
  'hu': 'hungarian',
  'ta': 'tamil',
  'no': 'norwegian',
  'th': 'thai',
  'ur': 'urdu',
  'hr': 'croatian',
  'bg': 'bulgarian',
  'lt': 'lithuanian',
  'la': 'latin',
  'mi': 'maori',
  'ml': 'malayalam',
  'cy': 'welsh',
  'sk': 'slovak',
  'te': 'telugu',
  'fa': 'persian',
  'lv': 'latvian',
  'bn': 'bengali',
  'sr': 'serbian',
  'az': 'azerbaijani',
  'sl': 'slovenian',
  'kn': 'kannada',
  'et': 'estonian',
  'mk': 'macedonian',
  'br': 'breton',
  'eu': 'basque',
  'is': 'icelandic',
  'hy': 'armenian',
  'ne': 'nepali',
  'mn': 'mongolian',
  'bs': 'bosnian',
  'kk': 'kazakh',
  'sq': 'albanian',
  'sw': 'swahili',
  'gl': 'galician',
  'mr': 'marathi',
  'pa': 'punjabi',
  'si': 'sinhala',
  'km': 'khmer',
  'sn': 'shona',
  'yo': 'yoruba',
  'so': 'somali',
  'af': 'afrikaans',
  'oc': 'occitan',
  'ka': 'georgian',
  'be': 'belarusian',
  'tg': 'tajik',
  'sd': 'sindhi',
  'gu': 'gujarati',
  'am': 'amharic',
  'yi': 'yiddish',
  'lo': 'lao',
  'uz': 'uzbek',
  'fo': 'faroese',
  'ht': 'haitian creole',
  'ps': 'pashto',
  'tk': 'turkmen',
  'nn': 'nynorsk',
  'mt': 'maltese',
  'sa': 'sanskrit',
  'lb': 'luxembourgish',
  'my': 'myanmar',
  'bo': 'tibetan',
  'tl': 'tagalog',
  'mg': 'malagasy',
  'as': 'assamese',
  'tt': 'tatar',
  'haw': 'hawaiian',
  'ln': 'lingala',
  'ha': 'hausa',
  'ba': 'bashkir',
  'jw': 'javanese',
  'su': 'sundanese',
}


def load_model(model_name: str = 'tiny.en') -> Model:
  # tokenizer = encoder()
  tokenizer = get_encoding('gpt2')
  state = torch.hub.load_state_dict_from_url(MODEL_URLS[model_name])
  state['dims']['n_vocab'] = tokenizer.n_vocab
  model = Model(state['dims'])
  model.load_state_dict(state['model_state_dict'])
  return model, tokenizer


def get_encoding(name: str):
  with open('data/gpt2.tiktoken', 'rb') as f:
    ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in f.read().splitlines())}
    n_vocab = len(ranks)
    specials = [
      '<|endoftext|>',
      '<|startoftranscript|>',
      *[f'<|{lang}|>' for lang in LANGUAGES.keys()],
      '<|translate|>',
      '<|transcribe|>',
      '<|startoflm|>',
      '<|startofprev|>',
      '<|nospeech|>',
      '<|notimestamps|>',
      *[f'<|{i * 0.02:.2f}|>' for i in range(1501)],
    ]

    special_tokens = dict(zip(specials, range(n_vocab, n_vocab + len(specials))))
    n_vocab += len(specials)

    return tiktoken.Encoding(
      name=name,
      explicit_n_vocab=n_vocab,
      pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
      mergeable_ranks=ranks,
      special_tokens=special_tokens,
    )


SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
SAMPLES_PER_SEGMENT = SAMPLE_RATE * 30
FRAMES_PER_SEGMENT = SAMPLES_PER_SEGMENT // HOP_LENGTH


def transcribe(model, tokenizer, audio, temperature=0.0):
  encoded_audio = model.encoder(audio.unsqueeze(0))
  tokens = torch.tensor([[tokenizer._special_tokens['<|startoftranscript|>']]], device=encoded_audio.device)
  eot = tokenizer._special_tokens['<|endoftext|>']

  max_tokens = model.decoder.n_text_ctx - len(tokens[0])

  for _ in range(max_tokens):
    with torch.no_grad():
      logits = model.decoder(tokens, encoded_audio)

    probs = F.softmax(logits[:, -1] / max(0.0001, temperature), dim=-1)

    if temperature <= 0:
      next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
    else:
      next_token = torch.multinomial(probs, num_samples=1)

    tokens = torch.cat([tokens, next_token], dim=-1)

    if next_token.item() == eot:
      break

  text = tokenizer.decode(tokens[0].tolist()).strip()
  return text


if __name__ == '__main__':
  model, tokenizer = load_model('tiny.en')
  audio = load_audio('data/sample.wav')
  text = transcribe(model, tokenizer, audio)
  print(text)
