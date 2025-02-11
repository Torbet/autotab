import math
import tiktoken
from typing import List, Set

import os
import requests
from bs4 import BeautifulSoup
import json
import re
import csv
import yt_dlp
import tempfile
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

    # get the urls if they exist
    self.url_file = os.path.join(self.scraping_data_dir, 'song_urls.txt')
    self.song_urls = []
    self.urls_exist = os.path.exists(self.url_file)
    if self.urls_exist:
      with open(self.url_file, 'r') as file:
        self.song_urls = [line.strip() for line in file]
        self.all_song_ids_set = set([re.search(r's(\d+)(?=t|\b)', url).group(1) for url in self.song_urls])

    # get the song dicts if they exist
    self.song_data_file = os.path.join(self.scraping_data_dir, 'song_apis.csv')
    self.song_api_urls = []
    if os.path.exists(self.song_data_file):
      with open(self.song_data_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)[1:]  # Read all rows into memory
        self.scraped_song_ids_set = set(row[0] for row in rows)
        self.song_api_urls = rows
        print(f'Loaded {len(self.song_api_urls)} song API URLs')
    else:
      with open(self.song_data_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['song_id', 'track_hash', 'video_api_url', 'tab_api_url'])
      self.scraped_song_ids_set = set()

    # make a file to store data on each track
    self.done_ids_path = os.path.join(self.data_dir, 'done_songs.csv')
    # If it doesn't exist write the header
    if not os.path.exists(self.done_ids_path):
      with open(self.done_ids_path, 'w') as file:
        pass
      self.done_ids = set()
    else:
      with open(self.done_ids_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        self.done_ids = set(row[0] for row in reader)

    self.model_data_path = os.path.join(self.data_dir, 'model_data.npz')

  def _safe_get(self, data, keys, default=None):
    """Safely get a nested value from a dictionary."""
    try:
      for key in keys:
        data = data[key]
      return data
    except Exception:
      return default

  def get_urls(self):
    """
    Read through paginated sitemaps to get all song URLs
    """
    if self.urls_exist:
      print(f"Song URLs already exist in '{self.url_file}'")
      return

    try:
      response = self.session.get('https://www.songsterr.com/sitemap-tabs.xml')
      response.raise_for_status()
    except Exception as e:
      raise (f'Failed to scrape main sitemap https://www.songsterr.com/sitemap-tabs.xml: {e}')

    soup = BeautifulSoup(response.content, 'lxml')
    page_urls = [sitemap.loc.text for sitemap in soup.find_all('sitemap')]

    for page_url in page_urls[:3]:
      try:
        response = self.session.get(page_url)
        response.raise_for_status()
      except Exception as e:
        print(f'Failed to scrape sitemap page {page_url}: {e}')
        continue
      soup = BeautifulSoup(response.content, 'lxml')
      self.song_urls.extend([url.loc.text for url in soup.find_all('url')])
      print(f'Found {len(self.song_urls)} song URLs')

    with open(self.url_file, 'w', encoding='utf-8') as file:
      for url in self.song_urls:
        file.write(url + '\n')
    print(f"{len(self.song_urls)} song URLs saved to '{self.url_file}'.")

  def get_song_data(self, start_from_start=False):
    """
    Scrape key data for each song
    """
    if not self.song_urls:
      print('No song URLs found. Run get_urls() first.')
      return

    with open(self.song_data_file, 'a', newline='') as file:
      writer = csv.writer(file)
      for url in self.song_urls:
        song_id = re.search(r's(\d+)(?=t|\b)', url).group(1)
        if song_id in self.scraped_song_ids_set:
          print('Scraped song already')
          continue

        # Get the big json object from the page
        try:
          response = self.session.get(url.strip())
          soup = BeautifulSoup(response.text, 'html.parser')
          json_string = soup.find('script', id='state').string
          data = json.loads(json_string)
        except Exception as e:
          print(f'Failed to scrape song at {url}: {e}')
          continue

        if not data:
          print(f'No data found for {url}')
          continue

        # get vaues
        revision_id = self._safe_get(data, ['meta', 'current', 'revisionId'])
        image = self._safe_get(data, ['meta', 'current', 'image'])
        name = re.sub(r'[\\/:*?"<>| ]', '_', self._safe_get(data, ['meta', 'current', 'title'], 'Unknown'))
        default_part_id = self._safe_get(data, ['meta', 'current', 'defaultTrack'], default=0)
        tracks = self._safe_get(data, ['meta', 'current', 'tracks'], default=None)
        default_hash = tracks[default_part_id]['hash'] if tracks else None

        # construct video url
        api_urls = [
          f'https://www.songsterr.com/api/video-points/{song_id}/{revision_id}/list',
          f'https://dqsljvtekg760.cloudfront.net/{song_id}/{revision_id}/{image}/{default_part_id}.json',
        ]

        # write the data to a json file and store it as a self.songs
        self.song_api_urls.append([song_id, default_hash] + api_urls)
        writer.writerow([song_id, default_hash] + api_urls)
        self.scraped_song_ids_set.add(song_id)
        print(f'Scraped {url}')

  def get_json(self, url):
    try:
      response = self.session.get(url)
      response.raise_for_status()
    except Exception as e:
      print(f'Failed to get JSON from {url}: {e}')
      return None
    return response.json()

  def download_temp_audio(self, video_id):
    """Download YouTube audio as an MP3 file."""
    fd, tmp_path = tempfile.mkstemp(suffix='.mp3')
    os.close(fd)  # Close the file descriptor; yt_dlp will create and write to this file.

    ydl_opts = {
      'format': 'bestaudio/best',
      'outtmpl': tmp_path,
      'postprocessors': [
        {
          'key': 'FFmpegExtractAudio',
          'preferredcodec': 'mp3',
          'preferredquality': '192',
        }
      ],
      'quiet': True,
      'no_warnings': True,
    }

    try:
      with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        url = f'https://www.youtube.com/watch?v={video_id}'
        ydl.download([url])
      print(f'Downloaded mp3 from {url}')
      return tmp_path
    except Exception:
      return None

  def download_audio_stream(self, video_id: str) -> np.ndarray:
    """
    Extracts the best audio URL from YouTube and uses ffmpeg to stream audio as PCM-wav.

    Returns:
        A numpy array containing the audio waveform.
    """
    SAMPLE_RATE = 16000
    video_url = f'https://www.youtube.com/watch?v={video_id}'

    # Configure yt-dlp to extract only audio formats
    ydl_opts = {
      'quiet': True,
      'format': 'bestaudio',  # Select best audio-only format
      'extract_flat': False,
      'format_sort': ['abr'],  # Sort by audio bitrate
    }

    # Extract video info and find the best audio format
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
      info = ydl.extract_info(video_url, download=False)

      # Get the URL from the selected format
      if 'url' in info:
        audio_url = info['url']
      elif 'formats' in info and len(info['formats']) > 0:
        # If direct URL not available, get it from the best format
        audio_url = info['formats'][0]['url']
      else:
        raise ValueError('Could not extract audio URL from any available format')

    # Use ffmpeg to read the audio URL and output raw PCM data in WAV format
    try:
      # Add user_agent and headers to avoid potential 403 errors
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
    MAX_TOKEN_SEG_LEN = 100
    enc = encoder()
    # for each song
    i = 0
    tabs = []
    audio = []
    seg_lens = []
    with open(self.done_ids_path, 'a', newline='') as file:
      writer = csv.writer(file)

      for song_id, track_hash, vid_api_url, tab_api_url in self.song_api_urls:
        if song_id in self.done_ids:
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
        try:
          waveform = self.download_audio_stream(video_id)
        except Exception as e:
          print(f'Error downloading audio: {e}')
          continue
        else:
          spectrogram_segments = process_audio_waveform(waveform, segment_audio_times)

        token_segments = [
          [bar_tokens[j] for j in range(segment_bar_indexes[i], segment_bar_indexes[i+1])]
          for i in range(len(segment_bar_indexes) - 1)
        ]

        self.done_ids.add(song_id)
        if len(token_segments) != len(spectrogram_segments):
          print(f'Incompatible segments: token segments: {len(token_segments)}, audio_segments: {len(spectrogram_segments)}')
          continue

        encoded_segments = self.encode_token_segments(enc, token_segments)

        seg_lens += [len(seg) for seg in encoded_segments]
        print("Cumulative mean token segment length:", np.mean(seg_lens))

        print("Number of segments: ", len(token_segments))
        for tab, aud in zip(encoded_segments, spectrogram_segments):
          if len(tab)< MAX_TOKEN_SEG_LEN:
            print(tab)
            tabs.append(np.pad(tab, (0, MAX_TOKEN_SEG_LEN - len(tab)), 'constant'))
            audio.append(aud)
        
        print("tabs.shape: ", tabs.shape)
        print("audio.shape: ", audio.shape)


        if i % 10 == 0:
          np.savez_compressed(self.model_data_path, tabs=tabs, audio=audio)
          print(f"Saved {len(tabs)} tabs and audio to file")

        writer.writerow(song_id)
      np.savez_compressed(self.model_data_path, tabs=tabs, audio=audio)

  