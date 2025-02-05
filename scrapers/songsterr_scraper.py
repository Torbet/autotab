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
        writer.writerow(['song_id', 'track_hash','video_api_url', 'tab_api_url'])
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
    fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
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
        'format_sort': ['abr']  # Sort by audio bitrate
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
            raise ValueError("Could not extract audio URL from any available format")
    
    # Use ffmpeg to read the audio URL and output raw PCM data in WAV format
    try:
        # Add user_agent and headers to avoid potential 403 errors
        out, _ = (
            ffmpeg
            .input(audio_url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=str(SAMPLE_RATE))
            .run(capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        print("ffmpeg error:", e)
        raise e
    except Exception as e:
        print("Error during ffmpeg processing:", str(e))
        raise e
    
    # Load the audio from the byte stream using soundfile
    audio_buffer = io.BytesIO(out)
    waveform, sr = sf.read(audio_buffer)
    
    if sr != SAMPLE_RATE:
        print(f"Warning: sample rate mismatch (expected {SAMPLE_RATE}, got {sr})")
    
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
            ((video['videoId'], video['points']) 
            for video in video_dicts 
            if video.get('tracks') and track_hash in video['tracks']),
            (default_video, default_points),  # Default values if no match is found])      
          )

  def get_segment_points(self, points):
    points = [p for p in points if p>=0]
    segments = [(i, t) for i, t in enumerate(points) if t >= (t // 30) * 30 and (i == 0 or t >= ((points[i-1] // 30) + 1) * 30)]
    return zip(*segments)

  def encode_token_segments(self, enc, token_segments):
    """
    Safely encode token segments with comprehensive error handling and validation.
    
    Args:
        enc: The encoder object
        token_segments: List of token lists to encode
    
    Returns:
        List of encoded segments
    """
    print(token_segments[0])
    unknown_tokens = self.validate_tokens_in_vocab(enc, token_segments)
    
    if unknown_tokens:
        raise ValueError(
            f"Found {len(unknown_tokens)} tokens not in vocabulary.\n"
            f"First few unknown tokens: {unknown_tokens[:10]}\n"
            "Please check your tokenization process."
        )
    encoded_segments = []
    
    for i, tokens in enumerate(token_segments):
        try:
            # Validate input
            if not tokens:
                print(f"Warning: Empty token list at index {i}")
                continue
                
            # Ensure all tokens are strings
            tokens = [str(t) for t in tokens]
            
            # Remove any empty or None tokens
            tokens = [t for t in tokens if t and t.strip()]
            
            if not tokens:
                print(f"Warning: All tokens were empty at index {i}")
                continue
            
            # Join with space and encode
            text = ' '.join(tokens)
            print(text)
            
            try:
                encoded = enc.encode(text)
                encoded_segments.append(encoded)
            except Exception as e:
                print(f"Encoding error at index {i}")
                print(f"Text being encoded: {repr(text)}")
                print(f"Tokens: {repr(tokens)}")
                print(f"Error: {str(e)}")
                raise
                
        except Exception as e:
            print(f"Unexpected error at index {i}")
            print(f"Token segment: {repr(tokens)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error: {str(e)}")
            
            # Re-raise with more context
            raise type(e)(f"Error processing segment {i}: {str(e)}") from e
            
    return encoded_segments
   
  def get_model_data(self):
    enc = encoder()
    # for each song
    i = 0
    data_pairs = []
    with open(self.done_ids_path, 'a', newline='') as file:
      writer = csv.writer(file)
      for song_id, track_hash, vid_api_url, tab_api_url in self.song_api_urls:
        if song_id in self.done_ids:
          continue
        i+=1
        print(f"\n{song_id}")

        video_dicts = self.get_json(vid_api_url)

        if not video_dicts:
          print("No video data")
          continue

        video_id, points = self.get_default_video(video_dicts, track_hash)
        
        tab_dict = self.get_json(tab_api_url)

        if not tab_dict:
          print("No tab data")
          continue

        tokens = tokenizer(tab_dict)
        bar_tokens = {i: t for i, t in tokens}

        segment_tab_indexes, segment_audio_times = self.get_segment_points(points)

        try:
          waveform = self.download_audio_stream(video_id)
        except Exception as e:
          print(f"Error downloading audio: {e}")
          continue
        else:
          spectrogram_segments = process_audio_waveform(waveform, segment_audio_times)
        print("Generated", len(spectrogram_segments), "spectrogram segments")

        segment_tab_indexes = list(segment_tab_indexes) + [len(bar_tokens)]
        segment_tab_indexes[0] = 1
        try:
          token_segments = [
            [bar_tokens[j+1] for j in range(segment_tab_indexes[i], segment_tab_indexes[i+1])] 
            for i in range(len(segment_tab_indexes)-1)]
        except Exception as e:
          print(f"Invalid tab index: {e}")
          print(segment_tab_indexes)
          print(bar_tokens.keys())
          continue

        self.done_ids.add(song_id)
        writer.writerow(song_id)
        if len(token_segments) != len(spectrogram_segments):
          print(f"Incompatible segments: token segments: {len(token_segments)}, audio_segments: {len(spectrogram_segments)}")
          continue
          
        # get the encoded tab data
        try:
          encoded_segments = self.encode_token_segments(enc, token_segments)
        except Exception as e:
          print("Failed to encode segments:")
          print(f"Error type: {type(e).__name__}")
          print(f"Error message: {str(e)}")        

        # encoded_segments = [enc.encode(' '.join(tokens)) for tokens in token_segments]
        data_pairs += zip(encoded_segments, spectrogram_segments)

        if i % 1000 == 0:
          np.savex(self.model_data_path, data_pairs)


      np.savex(self.model_data_path, data_pairs)
    
  def validate_tokens_in_vocab(self, enc: tiktoken.Encoding, token_segments: List[List[str]]) -> List[str]:
    """
    Check if all tokens are in the encoder's vocabulary.
    Returns a list of unknown tokens if any are found.
    
    Args:
        enc: The tiktoken encoder
        token_segments: List of token lists to validate
    
    Returns:
        List of tokens not found in vocabulary
    """
    unknown_tokens = set()
    
    # Get the full vocabulary for checking
    try:
        # Try to get the vocabulary directly if available
        vocab = set(enc.decode(list(range(enc.n_vocab))).split())
    except:
        # Fallback: reconstruct vocabulary from the encoder's internal state
        vocab = set()
        for token_bytes, _ in enc._mergeable_ranks.items():
            try:
                vocab.add(token_bytes.decode())
            except:
                continue
        vocab.update(enc._special_tokens.keys())
    
    # Check each token
    for i, tokens in enumerate(token_segments):
        for token in tokens:
            if not token:
                continue
                
            token_str = str(token).strip()
            if not token_str:
                continue
                
            try:
                # Try to encode the single token
                encoded = enc.encode(token_str)
                # Verify the token decodes back to itself
                decoded = enc.decode(encoded)
                if decoded.strip() != token_str:
                    unknown_tokens.add(token_str)
            except:
                unknown_tokens.add(token_str)
    
    if unknown_tokens:
        print(f"Found {len(unknown_tokens)} tokens not in vocabulary:")
        for token in sorted(unknown_tokens):
            print(f"  - '{token}'")
            
    return list(unknown_tokens)