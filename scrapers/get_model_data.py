import io
from tqdm import tqdm
import glob
import re
import csv
import gc
import numpy as np
import requests
import aiohttp
import asyncio
import yt_dlp
import ffmpeg
import soundfile as sf
from audio import process_audio_waveform, SAMPLE_RATE
from tokenizer import tokenizer, encoder


def get_last_batch_number(model_data_path_prefix: str) -> int:
  batch_files = glob.glob(f'{model_data_path_prefix}_batch_*.npz')
  batch_numbers = []
  for file in batch_files:
    match = re.search(rf'{re.escape(model_data_path_prefix)}_batch_(\d+)\.npz', file)
    if match:
      batch_numbers.append(int(match.group(1)))
  return max(batch_numbers) if batch_numbers else 0


async def get_json(url: str, session: aiohttp.ClientSession) -> dict:
  async with session.get(url) as response:
    if response.status != 200:
      raise RuntimeError(f'Failed to fetch {url}, status: {response.status}')
    try:
      return await response.json()
    except Exception as e:
      raise RuntimeError(f'Error decoding JSON from {url}: {e}') from e


def get_default_video(video_dicts: list, track_hash: str):
  default_video, default_points = video_dicts[-1].get('videoId'), video_dicts[-1].get('points')
  if not track_hash:
    return default_video, default_points
  return next(
    ((video.get('videoId'), video.get('points')) for video in video_dicts if video.get('tracks') and track_hash in video['tracks']),
    (default_video, default_points),
  )


def get_segment_points(points: list):
  """
  Given a list of timing points, each representing the start point of a bar,
  determine segmentation points such that each segment is <30s.
  timestamps to be used for audio segmentation as `waveform[timestamp[i] : timestamp[i+1] - 1]`
  eg:         [2, 3, 28, 30, 39, 59, 60]
  bar_indexes: ________  __________
  timestamps:  ^^^^^^^^^^^^  ^^^^^^^^^^
  bar indexes are one behind timestamps because timestamps indicate the start of the bar.
  i.e. bars[3:5] includes tabs from the start of bar 3 to the start of bar 6
      timestamps[3:6] includes audio from the start of bar 3 to the start of bar 6

  """
  # Only keep non-negative timing points.
  points = [p for p in points if p >= 0]
  if not points:
    return [], []

  bars, timestamps = [], []

  begin = points[0]
  timestamps.append(begin)
  for i, p in enumerate(points):
    if p <= begin + 30:
      continue
    else:
      timestamps.append(points[i - 1])
      bars.append(i - 2)
      begin = points[i - 1]

  return bars, timestamps


def encode_token_segments(enc, token_segments: list) -> list:
  """Encode each token segment using the given encoder."""
  encoded_segments = []
  for tokens in token_segments:
    encoded_tokens = []
    tokens = ['<|startoftab|>'] + tokens + ['<|endoftab|>']
    for t in tokens:
      encoded_tokens += enc.encode(t, allowed_special=enc.special_tokens_set)
    encoded_segments.append(encoded_tokens)
  return encoded_segments


def download_audio_stream(video_id: str) -> np.ndarray:
  video_url = f'https://www.youtube.com/watch?v={video_id}'
  ydl_opts = {
    'quiet': True,
    'format': 'bestaudio',
    'extract_flat': False,
    'format_sort': ['abr'],
    # 'cookiesfrombrowser': ('chrome', 'Default'),
    # 'extractor_args': 'youtube:po_token=web.gvs+Mltvcq-0xNFZR29KT7OyxKbpCrGn8OHnFzzXnwOYz2VLw0XHxBYpZZYyLkQH10yUjZhqcpxQ6QUfr1h3Cr2_HnfCrny2EVE2W-dPmOoNRwZCQXglMH_c6-HO4JrN'
  }
  with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    try:
      info = ydl.extract_info(video_url, download=False)
      if 'url' in info:
        audio_url = info['url']
      elif 'formats' in info and len(info['formats']) > 0:
        audio_url = info['formats'][0]['url']
      else:
        raise ValueError(f'Could not extract audio URL for video {video_id} from any available format')
    except Exception as e:
      raise e

  try:
    out, _ = (
      ffmpeg.input(audio_url)
      .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=str(SAMPLE_RATE))
      .run(capture_stdout=True, capture_stderr=True)
    )
  except Exception as e:
    raise RuntimeError(f'Error during ffmpeg processing for video {video_id}: {e}') from e

  audio_buffer = io.BytesIO(out)
  try:
    waveform, sr = sf.read(audio_buffer)
  except Exception as e:
    raise RuntimeError(f'Error reading audio data for video {video_id}: {e}') from e

  if sr != SAMPLE_RATE:
    print(f'Warning: sample rate mismatch (expected {SAMPLE_RATE}, got {sr})')
  return waveform


def get_vocab(enc):
  vocab = set()
  for token_bytes, _ in enc._mergeable_ranks.items():
    vocab.add(token_bytes.decode())
  vocab.update(enc._special_tokens.keys())
  return vocab


def raise_unknown_tokens(enc, bar_tokens, vocab):
  unknown_tokens = set()
  pattern = r'<[^>]+>|[^\s<]+|\s+'
  tokens_set = {t for token in bar_tokens for t in re.findall(pattern, str(token).strip())}
  unknown_tokens = tokens_set - vocab
  if unknown_tokens:
    raise Exception(f'Found {len(unknown_tokens)} tokens not in vocabulary:\n' + '\n'.join(f'  - {token}' for token in sorted(unknown_tokens)))


def chunker(seq, size):
  """Yield successive chunks of size 'size' from the sequence 'seq'."""
  for pos in range(0, len(seq), size):
    yield seq[pos : pos + size]


def read_checkpoint(checkpoint_file):
  try:
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
      return int(f.read().strip())
  except FileNotFoundError:
    # create the file if it doesn't exist
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
      f.write('0')
    return 0  # Start at the beginning if no checkpoint is found


def write_checkpoint(index, checkpoint_file):
  with open(checkpoint_file, 'w', encoding='utf-8') as f:
    f.write(str(index))


def get_bar_dict(tokens):
  """
  Convert tokens of form [(bar, tokA),(bar, tokB)...] into dict {bar: [tokA, tokB]...a}
  """
  bar_dict = {}
  for bar_num, bar_tokens in tokens:
    # zero index bars
    bar_num -= 1
    if bar_num in bar_dict:
      bar_dict[bar_num].append(bar_tokens)
    else:
      bar_dict[bar_num] = [bar_tokens]
  return bar_dict


async def process_song(song, enc, vocab, session, max_token_seg_len, batch_pbar):
  song_id, track_hash, vid_api_url, tab_api_url = song
  batch_pbar.write(f'\nProcessing Song ID: {song_id}')

  # Retrieve video and tab data concurrently.
  video_task = get_json(vid_api_url, session)
  tab_task = get_json(tab_api_url, session)
  video_dicts, tab_dict = await asyncio.gather(video_task, tab_task)

  if not video_dicts:
    raise RuntimeError(f'No video data returned for song {song_id}')
  video_id, points = get_default_video(video_dicts, track_hash)
  if not video_id or not points:
    raise RuntimeError(f'No valid video found for song {song_id}')
  if not tab_dict:
    raise RuntimeError(f'No tab data returned for song {song_id}')

  tokens = tokenizer(tab_dict)
  tokens_list = [t for _, t in tokens]
  raise_unknown_tokens(enc, tokens_list, vocab)
  bar_dict = get_bar_dict(tokens)
  bar_slice_indices, timestamps = get_segment_points(points)
  num_bars = len(points)
  bar_slice_indices.append(num_bars)

  # Build token segments.
  token_segments = []
  start_bar = 0
  for end_bar in bar_slice_indices:
    segment_tokens = [token for bar in range(start_bar, end_bar + 1) for token in bar_dict.get(bar, [])]
    token_segments.append(segment_tokens)
    start_bar = end_bar + 1

  encoded_segments = encode_token_segments(enc, token_segments)
  seg_lens = [len(seg) for seg in encoded_segments]

  # run in a thread so that the event loop isn’t blocked.
  waveform = await asyncio.to_thread(download_audio_stream, video_id)
  spectrogram_segments = await asyncio.to_thread(process_audio_waveform, waveform, timestamps)

  valid_segments = [
    (np.pad(tab_seg, (0, max_token_seg_len - len(tab_seg)), 'constant'), aud_seg)
    for tab_seg, aud_seg in zip(encoded_segments, spectrogram_segments)
    if len(tab_seg) < max_token_seg_len
  ]

  return song_id, valid_segments, seg_lens


async def get_model_data(
  song_meta_data: list,
  checkpoint_file: str,
  model_data_path_prefix: str,
  max_token_seg_len: int = 1000,
  batch_size: int = 10,
  num_batches: int = 1,
) -> None:
  enc = encoder()
  vocab = get_vocab(enc)
  checkpoint_index = read_checkpoint(checkpoint_file)
  batch_counter = get_last_batch_number(model_data_path_prefix)
  batch_ids = []
  song_id = None

  print(f'Batch size: {batch_size}')
  print(f'Num batches: {num_batches}')
  print(f"""Resuming from 
          song index: {checkpoint_index}
          song ID: {song_meta_data[checkpoint_index][0]}
          batch number: {batch_counter}
          """)
  print(f'Total number of songs {len(song_meta_data)}')
  song_meta_data = song_meta_data[checkpoint_index + 1 :]
  print(f'Number of unprocessed songs {len(song_meta_data)}')
  gc.collect()

  for batch in chunker(song_meta_data, batch_size):
    batch_tabs = []
    batch_audio = []

    async with aiohttp.ClientSession() as session:
      with tqdm(total=len(batch), desc=f'Processing Batch {batch_counter + 1}', unit='song') as batch_pbar:
        for song in batch:
          song_id = song[0]
          try:
            processed_song = await asyncio.wait_for(process_song(song, enc, vocab, session, max_token_seg_len, batch_pbar), timeout=30)
            song_id, valid_segments, seg_lens = processed_song

            if valid_segments:
              tabs, audio = zip(*valid_segments)
              batch_tabs.extend(tabs)
              batch_audio.extend(audio)
            batch_pbar.write(f'Valid segments: {len(valid_segments)}')

          except asyncio.TimeoutError:
            batch_pbar.write(f'Skipping song {song_id} due to timeout.')
          except Exception as e:
            batch_pbar.write(f'Failed processing song {song_id}: {e}')
          finally:
            batch_ids.append(song_id)
            batch_pbar.update(1)

    # Save and clear batch data.
    batch_counter += 1
    batch_filename = f'{model_data_path_prefix}_batch_{batch_counter}.npz'
    np.savez_compressed(batch_filename, tabs=np.array(batch_tabs, dtype=np.uint16), audio=np.array(batch_audio, dtype=np.float32))
    tqdm.write(f'Saved batch {batch_counter}: {len(batch_tabs)} tabs and {len(batch_audio)} audio segments to {batch_filename}')
    write_checkpoint(song_id, checkpoint_file)

    gc.collect()

    if batch_counter >= num_batches:
      break

  if batch_tabs and batch_audio:
    batch_counter += 1
    batch_filename = f'{model_data_path_prefix}_batch_{batch_counter}.npz'
    np.savez_compressed(batch_filename, tabs=np.array(batch_tabs), audio=np.array(batch_audio))
    print(f'Saved final batch {batch_counter}: {len(batch_tabs)} tabs and {len(batch_audio)} audio segments to {batch_filename}')
