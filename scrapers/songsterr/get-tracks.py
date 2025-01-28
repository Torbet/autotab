import requests
import csv
from bs4 import BeautifulSoup
import json

songs = []
session = requests.Session()

with open('song_urls.txt', 'r') as file:
    ids = []
    for line in file:
        url = line.strip()
        try:
            response = session.get(url)
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            continue
        if response.status_code != 200:
            print(f"{response.status_code} for {url}")
            continue
        soup = BeautifulSoup(response.text, 'html.parser')
        json_string = soup.find('script', id='state').string
        if not json_string:
            continue
        data = json.loads(json_string)
        song_id = data['route']['params']['songId']
        if song_id in ids:
            continue
        ids.append(song_id)
        name = data['meta']['current']['title']
        # clean name
        name = name.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('?', '_').replace(':', '_').replace('*', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')

        original_tracks = data['meta']['originalTracks']
        tracks = data['meta']['current']['tracks']
        try:
            prev_revision_id = data['meta']['current']['prevRevisionId']
        except KeyError:
            prev_revision_id = None
        revision_id = data['meta']['current']['revisionId']
        image = data['meta']['current']['image']


        song_data = {
            'song_id': song_id,
            'revision_id': revision_id,
            'prev_revision_id': prev_revision_id,
            'image': image,
            'name': name,
            'tracks': tracks,
            'original_tracks': original_tracks
        }

        # create a json file for each song
        print(f"Scraped {url}")
        with open('data/songsterr-data/songs/' + name + '.json', 'w') as file:
            json.dump(song_data, file, indent=4)



        


