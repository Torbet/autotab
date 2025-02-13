import os
import requests
from bs4 import BeautifulSoup
import json
import re
import csv
import yt_dlp
import io
import ffmpeg
import soundfile as sf

import numpy as np

from audio import process_audio_waveform
from tokenizer import tokenizer, encoder


class SongsterrScraper:
  def __init__(self):
    self.session = requests.Session()
    self.session.headers.update(
      {
        'User-Agent': 'Mediapartners-Google*',  # Pretend to be Googlebot
      }
    )

    # Dir to store model data
    self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(self.data_dir, exist_ok=True)

    # Dir to store scraped data
    self.scraping_data_dir = os.path.join(self.data_dir, 'songsterr-data')
    os.makedirs(self.scraping_data_dir, exist_ok=True)

    # make a file to store data on each track
    self.done_ids_path = os.path.join(self.data_dir, 'done_songs.txt')
    # If it doesn't exist write the header
    if not os.path.exists(self.done_ids_path):
      with open(self.done_ids_path, 'w') as file:
        pass
      self.done_ids = set()
    else:
      with open(self.done_ids_path, 'r') as file:
        self.done_ids = set([int(line.strip()) for line in file])

    self.model_data_path = os.path.join(self.data_dir, 'model_data.npz')

    def download_audio_stream(self, video_id: str) -> np.ndarray:
    SAMPLE_RATE = 16000
    video_url = f'https://www.youtube.com/watch?v={video_id}'

    ydl_opts = {
      'quiet': True,
      'format': 'bestaudio',  # Select best audio-only format
      'extract_flat': False,
      'format_sort': ['abr'],  # Sort by audio bitrate
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
      info = ydl.extract_info(video_url, download=False)
      if 'url' in info:
        audio_url = info['url']
      elif 'formats' in info and len(info['formats']) > 0:
        audio_url = info['formats'][0]['url']
      else:
        raise ValueError('Could not extract audio URL from any available format')
    try:
      out, _ = (
        ffmpeg.input(
          audio_url,
          headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
          },
        )
        .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=str(SAMPLE_RATE))
        .run(capture_stdout=True, capture_stderr=True)
      )
    except Exception as e:
      print('ffmpeg error:', e)
      raise e
    except Exception as e:
      print('Error during ffmpeg processing:', str(e))
      raise e

    # Load the audio from the byte stream using soundfile
    audio_buffer = io.BytesIO(out)
    waveform, sr = sf.read(audio_buffer)

    if sr != SAMPLE_RATE:
      print(f'Warning: sample rate mismatch (expected {SAMPLE_RATE}, got {sr})')

    return waveform

  def get_default_video(self, video_dicts, track_hash):
    default_video, default_points = next(
      ((video['videoId'], video['points']) for video in video_dicts if video['feature'] != 'alternative'),
      (None, None),  # Default values if no match is found
    )
    if not track_hash:
      return default_video, default_points
    else:
      return next(
        ((video['videoId'], video['points']) for video in video_dicts if video.get('tracks') and track_hash in video['tracks']),
        (default_video, default_points),  # Default values if no match is found])
      )

  def get_segment_points(self, points):
    points = [p for p in points if p >= 0]
    segs = list(range(1, (int(points[-1]) // 30)+1))
    segs.append(999999)
    bar_indexes = []
    timestamps = []
    j = 0
    for i, t in enumerate(points):
      if t >= 30 * segs[j]:
        bar_indexes.append(i-2)
        timestamps.append(points[i-2])
        j+=1
    return bar_indexes, timestamps

  def encode_token_segments(self, enc, token_segments):
    encoded_segments = []

    for i, tokens in enumerate(token_segments):
      encoded_tokens = []
      for t in tokens:
        encoded_tokens += enc.encode(t)
      encoded_segments.append(encoded_tokens)

    return encoded_segments

  def get_model_data(self):
    MAX_TOKEN_SEG_LEN = 1000
    enc = encoder()
    # for each song
    i = 0
    tabs = []
    audio = []
    seg_lens = []
    for song_id, track_hash, vid_api_url, tab_api_url in self.song_api_urls:
      try:
        if int(song_id) in self.done_ids:
          print("Skipping... already scraped")
          continue
        i += 1
        print(f'\nSong ID: {song_id}')

        video_dicts = self.get_json(vid_api_url)

        if not video_dicts:
          print('No video data')
          continue

        video_id, points = self.get_default_video(video_dicts, track_hash)

        tab_dict = self.get_json(tab_api_url)

        if not tab_dict:
          print('No tab data')
          continue

        tokens = tokenizer(tab_dict)
        bar_tokens = [t for _, t in tokens]

        segment_bar_indexes, segment_audio_times = self.get_segment_points(points)
        segment_bar_indexes = list(segment_bar_indexes) + [len(bar_tokens)-1]
        waveform = self.download_audio_stream(video_id)
        spectrogram_segments = process_audio_waveform(waveform, segment_audio_times)

        token_segments = [
          [bar_tokens[j] for j in range(segment_bar_indexes[i], segment_bar_indexes[i+1])]
          for i in range(len(segment_bar_indexes) - 1)
        ]

        encoded_segments = self.encode_token_segments(enc, token_segments)

        seg_lens += [len(seg) for seg in encoded_segments]
        print("Cumulative mean token segment length:", np.mean(seg_lens))

        print("Number of segments in song: ", len(token_segments))

        for tab, aud in zip(encoded_segments, spectrogram_segments):
          if len(tab)< MAX_TOKEN_SEG_LEN:
            tabs.append(np.pad(tab, (0, MAX_TOKEN_SEG_LEN - len(tab)), 'constant'))
            audio.append(aud)
        
        print("tabs.shape: ", np.array(tabs).shape)
        print("audio.shape: ", np.array(audio).shape)

        if i % 10 == 0:
          np.savez_compressed(self.model_data_path, tabs=tabs, audio=audio)
          print(f"Saved {len(tabs)} tabs and audio to file")

      except Exception as e:
        print(f"Failed: {e}")
      with open(self.done_ids_path, 'a', encoding='utf-8') as file:
        file.write(song_id + '\n')
    np.savez_compressed(self.model_data_path, tabs=tabs, audio=audio)

  