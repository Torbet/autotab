import tiktoken
import json
import requests
import gzip
import urllib.request


def tokenizer(track):
  # with open(track, 'r') as file:
  #   track = json.load(file)

  tokens = []
  bars = track['measures']

  for i in range(len(bars)):
    beat = bars[i]['voices'][0]['beats']
    for x in range(len(beat)):
      notes = beat[x].get('notes', [])
      # duration = beat[x].get('duration', None)

      # if beat[x].get('rest', False):
      #   if duration:
      #     num = duration[0]
      #     den = duration[1]
      #     dur = round_time(num, den)
      #     tokens.append((i + 1, f'<R><T{dur}>'))
      note_tokens = []
      if len(notes) > 1:
        note_tokens.append('<C>')
      for note in range(len(notes)):
        fret = notes[note].get('fret', None)
        string = notes[note].get('string', None)
        # tie = notes[note].get('tie', None)
        # bend = notes[note].get('bend', None)
        hp = notes[note].get('hp', None)
        if fret is not None and string is not None:
          fret_str = f'F{fret}'
          str_str = f'S{string + 1}'
          note_tokens.append(f'<{str_str}><{fret_str}>')
          if hp is not None:
            if x < len(beat) - 1:
              next_note = beat[x + 1]['notes'][0]
            elif i < len(bars) - 1:
              next_note = bars[i + 1]['voices'][0]['beats'][0]['notes'][0]
            else:
              next_note = beat[x]['notes'][0]

            if 'fret' in next_note and fret < next_note['fret']:
              note_tokens.append('<H>')
            else:
              note_tokens.append('<P>')
      if len(notes) > 1:
        note_tokens.append('</C>')
      tokens.append((i + 1, ''.join(note_tokens)))

  return tokens
  # if tie is not None:
  #   note_tokens.append('<TI>')
  # if bend is not None:
  #   note_tokens.append('<B>')
  # if hp is not None:
  #   if x < len(beat) - 1:
  #     next_note = beat[x + 1]['notes'][0]
  #   elif i < len(bars) - 1:
  #     next_note = bars[i + 1]['voices'][0]['beats'][0]['notes'][0]
  #   else:
  #     next_note = beat[x]['notes'][0]

  #   if 'fret' in next_note and fret < next_note['fret']:
  #     note_tokens.append('<H>')
  #   else:
  #     note_tokens.append('<P>')

  # if note_tokens and duration:
  #   num = duration[0]
  #   den = duration[1]
  #   dur = round_time(num, den)
  #   combined_note_token = ''.join(note_tokens) + f'<T{dur}>'

  # if beat[x].get('letRing', False):
  #   combined_note_token += '<LR>'
  # if beat[x].get('upStroke', False):
  #   combined_note_token += '<US>'
  # if beat[x].get('slide', False):
  #   # combined_note_token += f'<SL{beat[x]["slide"]}>'
  #   combined_note_token += '<SL>'
  # if len(notes) > 1:

  # tokens.append((i + 1, combined_note_token))


def encoder():
  tokens = ['<PAD>']
  tokens.extend([f'<S{s}><F{f}>' for s in range(1, 7) for f in range(-1, 21)])
  tokens.extend(['<C>', '</C>'])
  special = ['<|startoftab|>', '<|endoftab|>']
  ranks = {token.encode(): i for i, token in enumerate(tokens)}
  special = {token: len(ranks) + i for i, token in enumerate(special)}
  n_vocab = len(ranks) + len(special)
  return tiktoken.Encoding(name='tab', explicit_n_vocab=n_vocab, pat_str=r'<[^>]+>|[^\s<]+|\s+', mergeable_ranks=ranks, special_tokens=special)


def validate_tokens_in_vocab(enc, tokens):
  unknown_tokens = set()

  vocab = set()
  for token_bytes, _ in enc._mergeable_ranks.items():
    try:
      vocab.add(token_bytes.decode())
    except:
      continue
  vocab.update(enc._special_tokens.keys())

  # Check each token
  for token in tokens:
    if not token:
      continue

    token_str = str(token).strip()
    if not token_str:
      continue

    try:
      # Try to encode the single token
      encoded = enc.encode(token_str)
      # Verify the token decodes back to itself
      decoded = enc.decode(encoded)
      if decoded.strip() != token_str:
        unknown_tokens.add(token_str)
    except:
      unknown_tokens.add(token_str)

  if unknown_tokens:
    print(f'Found {len(unknown_tokens)} tokens not in vocabulary:')
    for token in sorted(unknown_tokens):
      print(f"  - '{token}'")

  return list(unknown_tokens)


# def round_time(x, y):
#   value = x / y

#   best_match = min(valid_times, key=lambda d: abs(value - (1 / d)))

#   return best_match


# path = 'data/tabs_jsons/1100_2.json'

# path = 'data/songsterr-data/Superman_0.json'

tokenurl = 'https://dqsljvtekg760.cloudfront.net/103/1017529/v3-5-24-ipkd1DcEtxBtNp23/0.json'

r = requests.get(tokenurl)

tokens = tokenizer(r.json())

# tokens = tokenizer(path)
# tokens = [t for _, t in tokens]

for t in tokens:
  print(t)

# # # tokens = tokenizer(tab_dict)
# # # #tokens = [token for i, token, in tokens]
# # # #print(len(tokens))
# enc = encoder()
# validate_tokens_in_vocab(enc, tokens)
