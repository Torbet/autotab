import csv
import json
import re
from typing import List
import requests
from bs4 import BeautifulSoup

from scrapers.utils import safe_get, load_scraped_song_ids

def get_song_meta_data(
    song_urls: List[str],
    song_meta_data_file: str,
    session: requests.Session,
    num_songs: int = 10000000,
    start_from_start: bool = False,
) -> List[List[str]]:
    """
    Scrape key song meta data from each URL and append API information to a CSV file.
    """
    if not song_urls:
        print('No song URLs found.')
        return []
    
    print(f"Beginning scraping, stopping after {num_songs}")

    scraped_song_ids_set = load_scraped_song_ids(song_meta_data_file)
    song_id_pattern = re.compile(r's(\d+)(?=t|\b)')

    num_songs_scraped = len(scraped_song_ids_set)
    # Open CSV file in append mode
    with open(song_meta_data_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for url in (u.strip() for u in song_urls):
            if num_songs_scraped >= num_songs:
                print(f"Stopping... Scraped {num_songs}")
                return
            # get song id from url
            if not (match := song_id_pattern.search(url)):
                print(f"Could not extract song ID from URL: {url}")
                continue
            song_id = match.group(1)

            # Chekc if song has been scraped
            if song_id in scraped_song_ids_set:
                print(f'Scraped song {song_id} already')
                continue

            # scrape metadata
            try:
                response = session.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                state_script = soup.find('script', id='state')
                if not state_script or not state_script.string:
                    print(f'No state script found for {url}')
                    continue
                data = json.loads(state_script.string)
            except Exception as e:
                print(f'Failed to scrape song at {url}: {e}')
                continue

            if not data:
                print(f'No data found for {url}')
                continue

            revision_id = safe_get(data, ['meta', 'current', 'revisionId'])
            image = safe_get(data, ['meta', 'current', 'image'])
            default_part_id = safe_get(data, ['meta', 'current', 'defaultTrack'], default=0)
            tracks = safe_get(data, ['meta', 'current', 'tracks'], default=None)
            default_hash = tracks[default_part_id]['hash'] if tracks and default_part_id < len(tracks) else None

            # Construct API URLs for video points and tab data.
            api_urls = [
                f'https://www.songsterr.com/api/video-points/{song_id}/{revision_id}/list',
                f'https://dqsljvtekg760.cloudfront.net/{song_id}/{revision_id}/{image}/{default_part_id}.json'
            ]
            row = [song_id, default_hash] + api_urls

            # Append new song data to our list and CSV file.
            writer.writerow(row)
            scraped_song_ids_set.add(song_id)
            print(f'Scraped {song_id} at:\n {url}')