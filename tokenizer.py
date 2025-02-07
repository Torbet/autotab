import tiktoken
import json


def tokenizer(track):
  tokens = []
  bars = track['measures']

  for i in range(len(bars)):
    for beat in bars[i]['voices'][0]['beats']:
      notes = beat.get('notes', [])
      duration = beat.get('duration', None)

      if beat.get('rest', False):
        if duration:
          dur = f'T{duration}'
          tokens.append((i + 1, f'<R><{dur}>'))

      else:
        note_tokens = []
        for note in notes:
          fret = note.get('fret', None)
          string = note.get('string', None)
          tie = note.get('tie', None)
          bend = note.get('bend', None)
          hp = note.get('hp', None)
          if fret is not None and string is not None and duration:
            fret_str = f'F{fret}'
            str_str = f'S{string}'
            if tie is not None:
              note_tokens.append(f'<{str_str}><{fret_str}><Ti>')
            if bend is not None:
              bendPoints = ''
              points = bend['points']
              for p in points:
                pos = p.get('position', None)
                tone = p.get('tone', None)
                bendPoints += f'<P{pos}><Tn{tone}>'
              note_tokens.append(f'<{str_str}><{fret_str}><B>' + bendPoints)
              # if hp is not None:
              # if hammer
              # if pull
            else:
              note_tokens.append(f'<{str_str}><{fret_str}>')

        if note_tokens and duration:
          dur = f'T{duration}'
          combined_note_token = ''.join(note_tokens) + f'<{dur}>'

          if beat.get('letRing', False):
            combined_note_token += '<LR>'
          if beat.get('upStroke', False):
            combined_note_token += '<US>'
          if beat.get('slide', False):
            combined_note_token += f'<Sl-{beat["slide"]}>'

          tokens.append((i + 1, combined_note_token))

  return tokens


def encoder():
  tokens = []
  tokens.extend([f'<S{i}>' for i in range(1, 7)])
  tokens.extend([f'<F{i}>' for i in range(1, 25)])
  tokens.extend([f'<T{2**i}>' for i in range(6)])
  tokens.extend(['<H>', '<P>', '<S>', '<B>'])  # hammer on, pull off, slide, bend
  special = ['<|endoftext|>', '<|startoftab|>', '<|endoftab|>']
  special.extend([f'<U{i}>' for i in range(51861 - len(tokens))])
  ranks = {token.encode(): i for i, token in enumerate(tokens)}
  special = {token: len(ranks) + i for i, token in enumerate(special)}
  n_vocab = len(ranks) + len(special)
  return tiktoken.Encoding(name='tab', explicit_n_vocab=n_vocab, pat_str=r'<[^>]+>|[^\s<]+|\s+', mergeable_ranks=ranks, special_tokens=special)


def validate_tokens_in_vocab(enc, tokens):
  unknown_tokens = set()

  # Get the full vocabulary for checking
  try:
    # Try to get the vocabulary directly if available
    vocab = set(enc.decode(list(range(enc.n_vocab))).split())
  except:
    # Fallback: reconstruct vocabulary from the encoder's internal state
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


path = 'data/tabs_jsons/1100_2.json'

with open(path, 'r') as file:
  tab_dict = json.load(file)
tokens = tokenizer(tab_dict)
tokens = [token for i, token in tokens]
print(len(tokens))
enc = encoder()
validate_tokens_in_vocab(enc, tokens)
# for t in token:
#   print(t)
