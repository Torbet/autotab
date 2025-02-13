import tiktoken
import json
import requests
import gzip

def tokenizer(track):
  # with open(track, 'r') as file:
  #   track = json.load(file)

  tokens = []
  bars = track['measures']

  for i in range(len(bars)):
    beat = bars[i]['voices'][0]['beats']
    for x in range(len(beat)):
      notes = beat[x].get('notes', [])
      duration = beat[x].get('duration', None)

      if beat[x].get('rest', False):
        if duration:
          dur = f'T{duration}'
          tokens.append((i + 1, f'<R><{dur}>'))

      else:
        note_tokens = []
        if len(notes) > 1:
          note_tokens.append('<C>')
        for note in range(len(notes)):
          fret = notes[note].get('fret', None)
          string = notes[note].get('string', None)
          tie = notes[note].get('tie', None)
          bend = notes[note].get('bend', None)
          hp = notes[note].get('hp', None)
          if fret is not None and string is not None and duration:
            fret_str = f'F{fret}'
            str_str = f'S{string+1}'
            note_tokens.append(f'<{str_str}><{fret_str}>')
            if tie is not None:
              note_tokens.append('<TI>')
            if bend is not None:
              note_tokens.append('<B>')
            if hp is not None:
              if x < len(beat)-1:
                next_note = beat[x + 1]
              elif i < len(bars)-1:
                next_note = bars[i+1]['voices'][0]['beats'][0]['notes'][0]
              else:
                next_note = beat[x]
              
              if 'string' in next_note and string < next_note['string']:
                note_tokens.append('<H>')
              else:
                note_tokens.append('<P>')

        if note_tokens and duration:
          dur = f'T{duration}'
          combined_note_token = ''.join(note_tokens) + f'<{dur}>'

          if beat[x].get('letRing', False):
            combined_note_token += '<LR>'
          if beat[x].get('upStroke', False):
            combined_note_token += '<US>'
          if beat[x].get('slide', False):
            # combined_note_token += f'<SL{beat[x]["slide"]}>'
            combined_note_token += '<SL>'
          if len(notes) > 1:
            combined_note_token += '</C>'

          tokens.append((i + 1, combined_note_token))          
          # tokens.append((i + 1, '<|space|>'))          
          
  return tokens


def encoder():
  tokens = []
  tokens.extend([f'<S{i}>' for i in range(1, 7)])
  tokens.extend([f'<F{i}>' for i in range(-1, 25)])
  tokens.extend([f'<T[{i}, {j}]>' for i in range(1,65) for j in range(1,65) if i <= j])
  tokens.extend(['<H>', '<P>', '<SL>', '<B>', '<US>', '<LR>', '<TI>', '<TN>', '<R>', '<C>', '</C>'])  # hammer on, pull off, slide, bend
  special = ['<|endoftext|>', '<|startoftab|>', '<|endoftab|>', '<|space|>']
  special.extend([f'<U{i}>' for i in range(51861 - len(tokens))])
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

#path = 'data/tabs_jsons/1100_2.json'

path = 'data/songsterr-data/Superman_0.json'

url = "https://dqsljvtekg760.cloudfront.net/103/1017529/v3-5-24-ipkd1DcEtxBtNp23/0.json"

# tab_dict = requests.get(url).json()
# # with open(path, 'r') as file:
# #     tab_dict = json.load(file)
    
# tokens = tokenizer(tab_dict)
# tokens = [t for _, t in tokens]

# # # tokens = tokenizer(tab_dict)
# # # #tokens = [token for i, token, in tokens]
# # # #print(len(tokens))
# enc = encoder()
# validate_tokens_in_vocab(enc, tokens)
# for t in tokens:
#   print(t)