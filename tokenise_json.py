import json
import urllib.request
import gzip


def tokenise_json(json_url):
  with urllib.request.urlopen(json_url) as response:
    if response.info().get('Content-Encoding') == 'gzip':
      with gzip.GzipFile(fileobj=response) as gz:
        data = json.loads(gz.read().decode('utf-8'))
    else:
      data = json.loads(response.read().decode('utf-8'))

    # print(len(data["measures"]))

    bars = data['measures']
    # note_data = bars['voices'][0]['beats']

    for bar in bars:
      for beat in bar['voices'][0]['beats']:
        notes = beat.get('notes', [])
        duration = beat.get('duration', None)

        for note in notes:
          if note.get('rest', False):
            if duration:
              dur = f'T{duration}'
              print(f'<R><{dur}>')
          else:
            fret = note.get('fret', None)
            string = note.get('string', None)
            if fret is not None and string is not None and duration:
              fret_str = f'F{fret}'
              str_str = f'S{string}'
              dur_str = f'T{duration}'
              print(f'<{str_str}><{fret_str}><{dur_str}>')

    return data


url = 'https://dqsljvtekg760.cloudfront.net/23/1210084/v3-5-26-fQQUyvwVEOGu4bdA/2.json'

token = tokenise_json(url)
