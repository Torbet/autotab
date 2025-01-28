import requests
import polars as pl
import csv
import json
import os
from pytube import YouTube
from pydub import AudioSegment

# Set the top-level directory relative to the current script location
TOP_DIR = os.path.dirname(os.path.abspath(__file__))

def get_json(url):
    response = requests.get(url)
    return response.json()

def youtube_to_mp3(video_id, name, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(TOP_DIR, 'data', 'songsterr-data', 'audio')

    os.makedirs(output_dir, exist_ok=True)

    url = f"https://www.youtube.com/watch?v={video_id}"
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()

    audio_file_path = audio_stream.download(output_path=output_dir, filename=f"{name}-{video_id}.mp4")

    mp3_file_path = os.path.join(output_dir, name + ".mp3")
    audio = AudioSegment.from_file(audio_file_path)
    audio.export(mp3_file_path, format="mp3")

    # Clean up the intermediate file
    os.remove(audio_file_path)

csv_file_path = os.path.join(TOP_DIR, 'data', 'songsterr-data', 'tabs_with_audio.csv')
songs_dir = os.path.join(TOP_DIR, 'data', 'songsterr-data', 'songs')

os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "Song Name",
        "Song ID",
        "Revision ID",
        "Track Hash",
        "Video ID",
        "Video Points",
        "Audio",
        "Tab"
    ])

    for file_name in os.listdir(songs_dir):
        try:
            file_path = os.path.join(songs_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as json_file:
                song_data = json.load(json_file)

            video_list_url = (
                f"https://www.songsterr.com/api/video-points/"
                f"{song_data['song_id']}/{song_data['revision_id']}/list"
            )

            video_dicts = get_json(video_list_url)

            # Get the tab for each track
            guitar_tracks = [track for track in song_data['tracks'] if track['hash'].startswith('guitar')]
            if not guitar_tracks:
                continue

            for track in guitar_tracks:
                tab_url = (
                    f"https://dqsljvtekg760.cloudfront.net/{song_data['song_id']}/"
                    f"{song_data['revision_id']}/{song_data['image']}/{track['partId']}.json"
                )
                track_tab = get_json(tab_url)
                hash_ = track['hash']
                video_dicts = [video for video in video_dicts if video["trackHashes"] is not None]
                video_dicts = [video for video in video_dicts if hash_ in video["trackHashes"]]
                for i in range(len(video_dicts)):
                    youtube_to_mp3(video_dicts[i]['videoId'], song_data['name'])
                    writer.writerow([
                        song_data['name'],
                        song_data['song_id'],
                        song_data['revision_id'],
                        track['hash'],
                        video_dicts[i]['videoId'],
                        video_dicts[i]['points'],
                        json.dumps(track_tab)  # Convert dict to a JSON string for the CSV
                    ])
                    print(f"Scraped {song_data['name']}")
        except Exception as e:
            print(f"Failed to scrape {file_name}: {e}")
            continue