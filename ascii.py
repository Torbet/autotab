from tokenizer import tokenizer
import re


def tokens_to_ascii(tokens):
  # Initialize the tab lines for 6 strings
  tab_lines = ['-' * 32 for _ in range(6)]
  current_pos = 0
  max_chords_per_line = 32
  lines = []

  # Regular expression to match tokens
  token_pattern = re.compile(r'<(C|/C|S\d|F\d)>')
  matches = token_pattern.findall(tokens)

  chord = []
  string = None
  in_chord = False
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
            tab_lines[string - 1] = tab_lines[string - 1][:current_pos] + str(fret) + tab_lines[string - 1][current_pos + 1 :]
      current_pos += 2
      if current_pos % max_chords_per_line == 0:
        lines.append('\n'.join(tab_lines))
        tab_lines = ['-' * 32 for _ in range(6)]
        current_pos = 0
      in_chord = False
    elif match.startswith('S'):
      string = int(match[1])
    elif match.startswith('F'):
      fret = int(match[1])
      if string is not None and 1 <= string < 7:
        if in_chord:
          chord.append((string, fret))
        else:
          if current_pos < len(tab_lines[string - 1]):
            tab_lines[string - 1] = tab_lines[string - 1][:current_pos] + str(fret) + tab_lines[string - 1][current_pos + 1 :]
          current_pos += 2
          if current_pos % max_chords_per_line == 0:
            lines.append('\n'.join(tab_lines))
            tab_lines = ['-' * 32 for _ in range(6)]
            current_pos = 0

  # Add any remaining tab lines
  if current_pos > 0:
    lines.append('\n'.join(tab_lines))

  # Join all lines into a single string
  ascii_tab = '\n\n'.join(lines)
  return ascii_tab


# Example usage
path = 'data/songsterr-data/Superman_0.json'
tokens = tokenizer(path)

# tokens = '<S2><F2><C><S1><F2><S2><F3><S3><F2><S4><F0></C><S4><F8><S1><F9><C><S1><F2><S2><F3><S3><F2><S4><F0></C><S2><F0>'

# Convert tokens array to one long string of just tokens
tokens = [t for _, t in tokens]
tokens = ''.join(tokens)
print(tokens)

# Convert tokens to ASCII tab
ascii_tab = tokens_to_ascii(tokens)
print(ascii_tab)
