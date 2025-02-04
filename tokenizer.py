import tiktoken
import json

def tokenizer(track):
  with open(path, 'r') as file:
    track = json.load(file)
    tokens = []
    bars = track['measures']

    for i in range(len(bars)):
      for beat in bars[i]['voices'][0]['beats']:
        notes = beat.get('notes', [])
        duration = beat.get('duration', None)

        if beat.get('rest', False):
          if duration:
            dur = f'T{duration}'
            tokens.append(f'{i+1}, <R><{dur}>')

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

            tokens.append(f'{i+1}, {combined_note_token}')

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


path = 'data/tabs_jsons/Superman_1.json'
#path = 'data/tabs_jsons/With_You_0.json'

token = tokenizer(path)
for t in token:
  print(t)
