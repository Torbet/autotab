import requests
from tokenizer import tokenizer


def generate_ascii_tab(tab):
  """
  Generate an ASCII guitar tab from an array of tokens to turn it
  into a human-readable format
  """

  num_bars = tab[-1][0]

  ascii_bar = [['-' for _ in range(32)] for _ in range(6)]

  for i in range(num_bars):
    return True

  return True


def tokens_to_ascii(tokens):
  # Initialize the tab structure
  num_bars = max(tokens, key=lambda x: x[0])[0]
  tab = [[['- ' for _ in range(32)] for _ in range(6)] for _ in range(num_bars)]

  # Dictionary to keep track of the current position in each bar
  bar_positions = {i: 0 for i in range(1, num_bars + 1)}

  for bar, token in tokens:
    if '<R>' in token:
      # Handle rest token
      duration_start = token.index('<T') + 2
      duration_end = token.index('>', duration_start)
      duration = int(token[duration_start:duration_end])
      bar_positions[bar] += 32 // duration
      if bar_positions[bar] >= 32:
        bar_positions[bar] = 31  # Ensure position is within the valid range
      continue

    string_num = int(token[token.index('<S') + 2]) - 1
    fret_start = token.index('<F') + 2
    fret_end = token.index('>', fret_start)
    fret = token[fret_start:fret_end]

    # Find the duration token
    duration_start = token.index('<T') + 2
    duration_end = token.index('>', duration_start)
    duration = int(token[duration_start:duration_end])

    # Calculate the position in the bar
    position = bar_positions[bar]

    # Ensure position is within the valid range
    if position >= 32:
      position = 31

    # Place the fret number on the corresponding string and position
    tab[bar - 1][string_num][position] = fret

    # Update the position in the bar based on the duration
    bar_positions[bar] += 32 // duration
    if bar_positions[bar] >= 32:
      bar_positions[bar] = 31  # Ensure position is within the valid range

  # Convert the tab structure to a string
  tab_str = ''
  for bar in tab:
    for string in bar:
      tab_str += ''.join(string) + '\n'
    tab_str += '\n'

  return tab_str


# Example token set
# tokens = [(1, '<S4><F7><H><T4>'), (1, '<S4><F10><T4>'), (1, '<S3><F7><H><T16>'), (1, '<S3><F10><P><T8>'), (1, '<S3><F10><P><T8>'),
#   (2, '<S4><F7><H><T4>'), (2, '<S4><F10><T4>'), (2, '<S3><F7><H><T16>'), (2, '<S3><F10><P><T8>'), (2, '<S3><F10><P><T8>')]

tokenurl = 'https://dqsljvtekg760.cloudfront.net/103/1017529/v3-5-24-ipkd1DcEtxBtNp23/0.json'

r = requests.get(tokenurl)

tokens = tokenizer(r.json())

# print(tokens)

# Convert tokens to ASCII tab
ascii_tab = tokens_to_ascii(tokens)
print(ascii_tab)
