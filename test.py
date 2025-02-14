from scrapers.get_model_data import get_segment_points
from scrapers.get_model_data import download_audio_stream
from audio import process_audio_waveform, SAMPLE_RATE, SAMPLES_PER_SEGMENT

import numpy as np


points = [5, 24, 25, 31, 48, 51, 59, 60, 61, 62, 88, 89, 90]

tokens = [
  (1, 'one'),
  (1, 'one'),
  (2, 'one'),
  (2, 'one'),
  (3, 'two'),
  (4, 'two'),
  (5, 'two'),
  (5, 'two'),
  (6, 'two'),
  (6, 'two'),
  (6, 'two'),
  (7, 'three'),
  (7, 'three'),
  (8, 'three'),
  (9, 'three'),
  (9, 'three'),
  (10, 'three'),
  (10, 'three'),
  (10, 'three'),
  (11, 'three'),
  (11, 'three'),
  (12, 'four'),
  (12, 'four'),
  (13, 'four'),
]

correct_token_segments = [
  ['one', 'one', 'one', 'one'],
  ['two', 'two', 'two', 'two', 'two', 'two', 'two'],
  ['three', 'three', 'three', 'three', 'three', 'three', 'three', 'three', 'three', 'three'],
  ['four', 'four', 'four'],
]


bar_dict = {}
for bar_num, bar_tokens in tokens:
  # zero index bars
  bar_num -= 1
  if bar_num in bar_dict:
    bar_dict[bar_num].append(bar_tokens)
  else:
    bar_dict[bar_num] = [bar_tokens]

segment_indexes, timestamps = get_segment_points(points)
print(segment_indexes)
assert segment_indexes == [2, 6, 10]

num_bars = len(points)
segment_indexes.append(num_bars)

token_segments = []
start_bar = 0
for end_bar in segment_indexes:
  segment_tokens = []
  for bar in range(start_bar, end_bar + 1):
    segment_tokens.extend(bar_dict.get(bar, []))
  token_segments.append(segment_tokens)
  start_bar = end_bar + 1

token_segments = []
start_bar = 0
for end_bar in segment_indexes:
  segment_tokens = [token for bar in range(start_bar, end_bar + 1) for token in bar_dict.get(bar, [])]
  token_segments.append(segment_tokens)
  start_bar = end_bar + 1

# assert token_segments == correct_token_segments

"""
TESTING SPLITTING WAVEFORM
"""
test_points = [
  0.12,
  0.6,
  3.78,
  6.87,
  9.98,
  13.14,
  16.23,
  19.39,
  22.48,
  25.61,
  28.75,
  31.88,
  35.02,
  38.15,
  41.29,
  44.4,
  47.55,
  50.62,
  53.8,
  56.91,
  60.07,
  63.18,
  66.27,
  69.43,
  72.54,
  75.67,
  78.79,
  81.92,
  85.01,
  88.17,
  91.3,
  94.41,
  97.59,
  100.7,
  103.84,
  107,
  110.06,
  113.22,
  116.33,
  119.49,
  122.6,
  125.76,
  128.82,
  132.01,
  135.14,
  138.27,
  141.39,
  144.5,
  147.63,
  150.74,
  153.88,
  156.99,
  160.12,
  163.26,
  166.39,
  169.53,
  172.66,
  175.78,
  178.91,
  182,
  185.16,
  188.29,
  191.49,
  194.54,
  197.67,
  200.78,
  203.89,
  207.03,
  210.16,
  213.3,
  216.46,
  219.57,
  222.73,
  225.81,
  228.93,
  232.06,
  235.17,
  238.33,
  241.44,
  244.6,
  247.73,
  250.82,
  253.96,
  257.18,
  261.27,
]
test_id = 'JxPj3GAYYZ0'

waveform = download_audio_stream(test_id)

_, timestamps = get_segment_points(test_points)

# Taken from audio.py - process_audio_waveform()
slice_indices = [int(t * SAMPLE_RATE) for t in timestamps]
slice_indices.append(len(waveform) - 1)
segments = []
start_slice = 0
for end_slice in slice_indices:
  segments.append(waveform[start_slice:end_slice])
  start_slice = end_slice + SAMPLE_RATE


print('segment lengths:')
for seg in segments:
  print(len(seg) / SAMPLE_RATE)
  assert len(seg) <= SAMPLES_PER_SEGMENT
segments = [np.pad(seg, (0, SAMPLES_PER_SEGMENT - len(seg)), 'constant') for seg in segments]
