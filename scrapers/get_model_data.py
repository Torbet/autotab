import io
from tqdm import tqdm
import glob
import re
import csv
import gc
import numpy as np
import requests
import yt_dlp
import ffmpeg
import soundfile as sf
from audio import process_audio_waveform, SAMPLE_RATE
from tokenizer import tokenizer, encoder


def get_last_batch_number(model_data_path_prefix: str) -> int:
    batch_files = glob.glob(f"{model_data_path_prefix}_batch_*.npz")
    batch_numbers = []
    for file in batch_files:
        match = re.search(rf"{re.escape(model_data_path_prefix)}_batch_(\d+)\.npz", file)
        if match:
            batch_numbers.append(int(match.group(1)))
    return max(batch_numbers) if batch_numbers else 0

def get_json(url: str, session: requests.Session) -> dict:
    response = session.get(url)
    response.raise_for_status()
    try:
        return response.json()
    except Exception as e:
        raise RuntimeError(f"Error decoding JSON from {url}: {e}") from e

def get_default_video(video_dicts: list, track_hash: str):
    default_video, default_points = next(
        ((video.get('videoId'), video.get('points')) 
         for video in video_dicts if video.get('feature') != 'alternative'),
        (None, None)
    )
    if not track_hash:
        return default_video, default_points
    return next(
        ((video.get('videoId'), video.get('points')) 
         for video in video_dicts if video.get('tracks') and track_hash in video['tracks']),
        (default_video, default_points)
    )

def get_segment_points(points: list):
    """
    Given a list of timing points, determine segmentation points such that each segment is <30s.
    Returns a tuple of (bar_indexes, timestamps) indicating token segmentation indices and corresponding audio timestamps.
    """
    # Only keep non-negative timing points.
    points = [p for p in points if p >= 0]
    if not points:
        return [], []
    
    timestamps = []
    bar_indexes = []
    # Start the first segment with the first timing point.
    seg_start = points[0]
    seg_end = seg_start + 30  # The upper bound (non-inclusive) for the current segment.
    
    # Iterate through the remaining points.
    for i, p in enumerate(points[1:]):
        i+=1
        # If the point is within the current segment duration, add it.
        if p < seg_end:
            continue
        else:
            # Otherwise, finish the current segment and start a new one with the current point.
            timestamps.append(points[i-2]) 
            bar_indexes.append(i-2)
            seg_start = points[i-1]
            seg_end = seg_start + 30
    
    return bar_indexes, timestamps


def encode_token_segments(enc, token_segments: list) -> list:
    """Encode each token segment using the given encoder."""
    encoded_segments = []
    for tokens in token_segments:
        encoded_tokens = []
        for t in tokens:
            encoded_tokens += enc.encode(t)
        encoded_segments.append(encoded_tokens)
    return encoded_segments

def download_audio_stream(video_id: str) -> np.ndarray:
    video_url = f'https://www.youtube.com/watch?v={video_id}'
    ydl_opts = {
        'quiet': True,
        'format': 'bestaudio',
        'extract_flat': False,
        'format_sort': ['abr'],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(video_url, download=False)
            if 'url' in info:
                audio_url = info['url']
            elif 'formats' in info and len(info['formats']) > 0:
                audio_url = info['formats'][0]['url']
            else:
                raise ValueError(f"Could not extract audio URL for video {video_id} from any available format")
        except Exception as e:
            raise e

    try:
        out, _ = (
            ffmpeg.input(audio_url)
            .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar=str(SAMPLE_RATE))
            .run(capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Error during ffmpeg processing for video {video_id}: {e}") from e

    audio_buffer = io.BytesIO(out)
    try:
        waveform, sr = sf.read(audio_buffer)
    except Exception as e:
        raise RuntimeError(f"Error reading audio data for video {video_id}: {e}") from e

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
        raise Exception(f"Found {len(unknown_tokens)} tokens not in vocabulary:\n" + \
"\n".join(f"  - {token}" for token in sorted(unknown_tokens)))

def chunker(seq, size):
    """Yield successive chunks of size 'size' from the sequence 'seq'."""
    for pos in range(0, len(seq), size):
        yield seq[pos:pos + size]

def read_checkpoint(checkpoint_file):
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return int(f.read().strip())
    except FileNotFoundError:
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
        bar_num-=1
        if bar_num in bar_dict:
            bar_dict[bar_num].append(bar_tokens)
        else:
            bar_dict[bar_num] = [bar_tokens]
    return bar_dict



def get_model_data(
    song_meta_data: list,
    checkpoint_file: str,
    model_data_path_prefix: str,
    session: requests.Session,
    max_token_seg_len: int = 1000,
    batch_size: int = 10
) -> None:
    enc = encoder()
    vocab = get_vocab(enc)
    checkpoint_index = read_checkpoint(checkpoint_file) 
    batch_tabs = []
    batch_audio = []
    seg_lens = []
    batch_counter = 0
    total_processed = 0
    batch_ids = []

     # Initialise the batch counter from existing files
    batch_counter = get_last_batch_number(model_data_path_prefix)
    print(f"Batch size: {batch_size}")
    print(f"""Resuming from 
          song index: {checkpoint_index}
          song ID: {song_meta_data[checkpoint_index][0]}
          batch number: {batch_counter}
          """)
    print(f"Total number of songs {len(song_meta_data)}")
    song_meta_data = song_meta_data[checkpoint_index+1:]
    print(f"Number of unproccessed songs {len(song_meta_data)}")
    # Assuming song_meta_data is a list and batch_size is defined
    for batch in chunker(song_meta_data, batch_size):
        # Create a new progress bar for this batch
        with tqdm(total=len(batch), desc=f"Processing Batch {batch_counter + 1}", unit="song") as batch_pbar:
            for song_id, track_hash, vid_api_url, tab_api_url in batch:
                num_valid_segments = 0
                try:
                    total_processed += 1
                    batch_pbar.write(f"\nProcessing Song ID: {song_id}")

                    # Retrieve video data
                    video_dicts = get_json(vid_api_url, session)
                    if not video_dicts:
                        raise RuntimeError(f"No video data returned for song {song_id}")
                    video_id, points = get_default_video(video_dicts, track_hash)
                    if not video_id or not points:
                        raise RuntimeError(f"No valid video found for song {song_id}")

                    # Retrieve tab data
                    tab_dict = get_json(tab_api_url, session)
                    if not tab_dict:
                        raise RuntimeError(f"No tab data returned for song {song_id}")

                    tokens = tokenizer(tab_dict)
                    tokens_list = [t for _, t in tokens]

                    # Check if tokens are in vocab
                    raise_unknown_tokens(enc, tokens_list, vocab)

                    bar_dict = get_bar_dict(tokens)

                    bar_slice_indices, timestamps = get_segment_points(points)

                    # Append the final index for token segmentation
                    num_bars = len(points)
                    bar_slice_indices.append(num_bars)

                    # Build token segments based on bar slice indexes
                    token_segments = []
                    start_bar = 0
                    for end_bar in bar_slice_indices:
                        segment_tokens = [token for bar in range(start_bar, end_bar + 1) for token in bar_dict.get(bar, [])]
                        token_segments.append(segment_tokens)
                        start_bar = end_bar + 1

                    encoded_segments = encode_token_segments(enc, token_segments)
                    seg_lens.extend(len(seg) for seg in encoded_segments)
                    batch_pbar.write("Cumulative mean token segment length: " + str(np.mean(seg_lens)))
                    batch_pbar.write("Number of segments in song: " + str(len(token_segments)))

                    waveform = download_audio_stream(video_id)
                    spectrogram_segments = process_audio_waveform(waveform, timestamps)
                    
                    valid_segments = [
                        (np.pad(tab_seg, (0, max_token_seg_len - len(tab_seg)), 'constant'), aud_seg)
                        for tab_seg, aud_seg in zip(encoded_segments, spectrogram_segments)
                        if len(tab_seg) < max_token_seg_len
                    ]

                    if valid_segments:
                        batch_tabs, batch_audio = map(list, zip(*valid_segments))
                    else:
                        batch_tabs, batch_audio = [], []

                    num_valid_segments = len(valid_segments)
                    batch_pbar.write("Number of valid segments: " + str(num_valid_segments))

                except Exception as e:
                    batch_pbar.write(f"Failed processing song {song_id}: {e}")
                finally:
                    # Mark this song as processed regardless of success or failure.
                    batch_ids.append(song_id)

                batch_pbar.update(1)

            # End of batch: save and clear batch data
            batch_counter += 1
            batch_filename = f"{model_data_path_prefix}_batch_{batch_counter}.npz"
            np.savez_compressed(batch_filename, tabs=np.array(batch_tabs), audio=np.array(batch_audio))
            tqdm.write(f"Saved batch {batch_counter}: {len(batch_tabs)} tabs and {len(batch_audio)} audio segments to {batch_filename}")
            
            write_checkpoint(total_processed, checkpoint_file)            

            batch_tabs.clear()
            batch_audio.clear()
            seg_lens.clear()
            gc.collect()    # Final save for any leftover data in the last batch

    if batch_tabs and batch_audio:
        batch_counter += 1
        batch_filename = f"{model_data_path_prefix}_batch_{batch_counter}.npz"
        np.savez_compressed(batch_filename, tabs=np.array(batch_tabs), audio=np.array(batch_audio))
        print(f"Saved final batch {batch_counter}: {len(batch_tabs)} tabs and {len(batch_audio)} audio segments to {batch_filename}")