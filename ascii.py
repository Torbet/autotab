from tokenizer import tokenizer
import re
import requests


def tokens_to_ascii(tokens):
  # Initialize the tab lines for 6 strings
  tab_lines = ['-' * 32 for _ in range(6)]
  current_pos = 0
  max_chords_per_line = 32
  lines = []

  # Regular expression to match tokens
  token_pattern = re.compile(r'<(C|/C|S\d|F\d+|H|P)>')
  matches = token_pattern.findall(tokens)

  chord = []
  string = None
  in_chord = False
  hammer_on = False
  pull_off = False
  for match in matches:
    if match == 'C':
      chord = []
      in_chord = True
    elif match == '/C':
      if len(chord) > 7:
        chord = chord[:7]
      for string, fret in chord:
        if 1 <= string < 7:
          if current_pos < len(tab_lines[string - 1]):
            tab_lines[string - 1] = tab_lines[string - 1][:current_pos] + str(fret) + tab_lines[string - 1][current_pos + len(str(fret)):]
      current_pos += max(len(str(fret)) for _, fret in chord) + 1
      if current_pos >= max_chords_per_line:
        lines.append('\n'.join(tab_lines))
        tab_lines = ['-' * 32 for _ in range(6)]
        current_pos = 0
      in_chord = False
    elif match.startswith('S'):
      string = int(match[1])
    elif match.startswith('F'):
      fret = int(match[1:])
      if string is not None and 1 <= string < 7:
        if in_chord:
          chord.append((string, fret))
        else:
          note = str(fret)
          if hammer_on:
            note += 'H'
            hammer_on = False
          elif pull_off:
            note += 'P'
            pull_off = False
          if current_pos < len(tab_lines[string - 1]):
            tab_lines[string - 1] = tab_lines[string - 1][:current_pos] + note + tab_lines[string - 1][current_pos + len(note):]
          current_pos += len(note) + 1
          if current_pos >= max_chords_per_line:
            lines.append('\n'.join(tab_lines))
            tab_lines = ['-' * 32 for _ in range(6)]
            current_pos = 0
    elif match == 'H':
      if string is not None and 1 <= string < 7:
        if current_pos > 0:
          tab_lines[string - 1] = tab_lines[string - 1][:current_pos - 1] + 'H' + tab_lines[string - 1][current_pos:]
    elif match == 'P':
      if string is not None and 1 <= string < 7:
        if current_pos > 0:
          tab_lines[string - 1] = tab_lines[string - 1][:current_pos - 1] + 'P' + tab_lines[string - 1][current_pos:]

  # Add any remaining tab lines
  if current_pos > 0:
    lines.append('\n'.join(tab_lines))

  # Join all lines into a single string
  ascii_tab = '\n\n'.join(lines)
  return ascii_tab


# Example usage
# path = 'data/songsterr-data/Superman_0.json'
# tokens = tokenizer(path)

# tokens = '<S2><F2><C><S1><F2><S2><F3><S3><F2><S4><F0></C><S4><F8><S1><F9><C><S1><F2><S2><F3><S3><F2><S4><F0></C><S2><F0>'

tokenurl = 'https://dqsljvtekg760.cloudfront.net/103/1017529/v3-5-24-ipkd1DcEtxBtNp23/0.json'

r = requests.get(tokenurl)

tokens = tokenizer(r.json())

# Convert tokens array to one long string of just tokens
tokens = [t for _, t in tokens]
tokens = ''.join(tokens)
print(tokens)

# Convert tokens to ASCII tab
ascii_tab = tokens_to_ascii(tokens)
print(ascii_tab)
