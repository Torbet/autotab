import json
import urllib.request
import gzip


def tokenise_json(path):
  with open(path, 'r') as file:
    data = json.load(file)

    tokens = []
    bars = data['measures']

    for bar in bars:
      for beat in bar['voices'][0]['beats']:
        notes = beat.get('notes', [])
        duration = beat.get('duration', None)


        if beat.get('rest', False):
          if duration:
            dur = f'T{duration}'
            tokens.append(f'<R><{dur}>')

        else:
          note_tokens = []
          for note in notes:
            fret = note.get('fret', None)
            string = note.get('string', None)
            tie = note.get('tie', None)
            bend = note.get('bend', None)
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
            if beat.get('hp', None):
              combined_note_token += '<HP>'

            tokens.append(combined_note_token)

    return tokens
  
  # let ring, upstroke, tie, bend, slide
  # hp = hammer/pull
  # duration is a fraction i.e [1, 8] is 1/8th
  # index is the bar number


#path = 'data/tabs_jsons/Superman_1.json'
path = 'data/tabs_jsons/With_You_0.json'

token = tokenise_json(path)
print(token)
