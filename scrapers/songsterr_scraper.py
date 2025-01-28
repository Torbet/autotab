from time import sleep
import os
import requests
from bs4 import BeautifulSoup
import json
import re
import csv
import yt_dlp


class SongsterrScraper:
    def __init__(self):

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mediapartners-Google*",  # Pretend to be Googlebot
        })

        # Dir to store model data
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        os.makedirs(self.data_dir, exist_ok=True)

        # Dir to store scraped data
        self.scraping_data_dir = os.path.join(self.data_dir, "songsterr-data")
        os.makedirs(self.scraping_data_dir, exist_ok=True)
        
        # get the urls if they exist
        self.url_file = os.path.join(self.scraping_data_dir, "song_urls.txt")
        self.song_urls = []
        self.urls_exist = os.path.exists(self.url_file) 
        if self.urls_exist:
            with open(self.url_file, 'r') as file:
                self.song_urls = [line.strip() for line in file]
                self.all_song_ids_set = set([re.search(r"s(\d+)(?=t|\b)", url).group(1) for url in self.song_urls])

        # get the song dicts if they exist
        self.song_data_dir = os.path.join(self.scraping_data_dir, "songs")
        self.songs = []
        self.songs_exist = os.path.exists(self.song_data_dir)
        if self.songs_exist:
            for filename in os.listdir(self.song_data_dir):
                if filename.endswith('.json'):  
                    with open(os.path.join(self.song_data_dir, filename), 'r') as f:
                        self.songs.append(json.load(f))
        else:
            os.makedirs(self.song_data_dir, exist_ok=True)

        self.scraped_song_ids_set = set(str(song['song_id']) for song in self.songs)

        # Dir to store audio mp3s files
        self.audio_dir = os.path.join(self.data_dir, "audio")
        os.makedirs(self.audio_dir, exist_ok=True)

        # Dir to store tab jsons
        self.tabs_dir = os.path.join(self.data_dir, "tabs_jsons")
        os.makedirs(self.tabs_dir, exist_ok=True)

        # make a file to store data on each track
        self.track_data_file = os.path.join(self.data_dir, "tabs_with_audio.csv")
        # If it doesn't exist write the header
        if not os.path.exists(self.track_data_file):
            with open(self.track_data_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    "song_name",
                    "song_id",
                    "revision_id",
                    "track_hash",
                    "video_id",
                    "video_points",
                    "audio_tab_filename"
                ])
        with open(self.track_data_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            self.audio_video_scraped_song_ids_set = set(row[1] for row in reader)
        

    def _safe_get(self, data, keys, default=None):
        """Safely get a nested value from a dictionary."""
        try:
            for key in keys:
                data = data[key]
            return  data
        except KeyError:
            return default


    def get_urls(self):
        """
        Read through paginated sitemaps to get all song URLs
        """
        if self.urls_exist:
            print(f"Song URLs already exist in '{self.url_file}'")
            return

        main_sitemap_url = "https://www.songsterr.com/sitemap-tabs.xml"
        
        try:
            response = self.session.get(main_sitemap_url)
            response.raise_for_status()
        except Exception as e:
            raise(f"Failed to scrape main_sitemap_url {main_sitemap_url}: {e}")

        soup = BeautifulSoup(response.content, 'lxml')
        page_urls = [sitemap.loc.text for sitemap in soup.find_all('sitemap')]

        for page_url in page_urls[:3]:
            try:
                response = self.session.get(page_url)
                response.raise_for_status()
            except:
                print(f"Failed to scrape sitemap page {page}: {e}")
                continue
            soup = BeautifulSoup(response.content, 'lxml')
            self.song_urls.extend([url.loc.text for url in soup.find_all('url')])
            print(f"Found {len(self.song_urls)} song URLs")
        
        with open(self.url_file, "w", encoding="utf-8") as file:
            for url in self.song_urls:
                file.write(url + "\n")

        print(f"{len(self.song_urls)} song URLs saved to '{self.url_file}'.")


    def get_song_data(self, start_from_start=False):
        """
        Scrape key data for each song 
        """
        if not self.song_urls:
            print("No song URLs found. Run get_urls() first.")
            return
        
        if self.songs_exist and not start_from_start:
            print(f"Song data already exists in '{self.song_data_dir}'")
            return
        

        for url in self.song_urls:
            song_id = re.search(r"s(\d+)(?=t|\b)", url).group(1)
            if song_id in self.scraped_song_ids_set and not start_from_start:
                continue 

            # Get the big json object from the page
            try:
                response = self.session.get(url.strip())
                soup = BeautifulSoup(response.text, 'html.parser')
                json_string = soup.find('script', id='state').string
                data = json.loads(json_string)
            except Exception as e:
                print(f"Failed to scrape song at {url}: {e}")
                continue
            
            # extract ids, tracks, etc from the json, return None if not found
            song_id = self._safe_get(data, ['route','params','songId'])
            if song_id in ids:
                continue
            ids.add(song_id)
            song_data = {
                'song_id': song_id,
                'revision_id': self._safe_get(data, ['meta', 'current', 'revisionId']),
                'prev_revision_id': self._safe_get(data, ['meta', 'current', 'prevRevisionId']),
                'image': self._safe_get(data, ['meta', 'current', 'image']),
                'name': re.sub(r'[\\/:*?"<>| ]', '_', self._safe_get(data, ['meta', 'current', 'title'], 'Unknown')),
                'tracks': self._safe_get(data, ['meta', 'current', 'tracks']),
                'original_tracks': self._safe_get(data, ['meta', 'originalTracks'])
            }
            
            # write the data to a json file and store it as a self.songs 
            self.songs.append(song_data)
            with open(os.path.join(self.song_data_dir, song_data['name'] + '.json'), 'w') as file:
                json.dump(song_data, file, indent=4)

            print(f"Scraped {url}")

    def get_audio_and_tab(self, start_from_start=False):
        video_ids = set()

        def get_json(url):
            try:
                response = requests.get(url)
                response.raise_for_status()
            except Exception as e:
                print(f"Failed to get JSON from {url}: {e}")
                return None
            return response.json()

        def download_youtube_audio(video_id, name):
            """Download YouTube audio as an MP3 file."""
            output_path = os.path.join(self.audio_dir, f"{name}.mp3")
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(output_path),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
            }
    
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    url = f"https://www.youtube.com/watch?v={video_id}"
                    ydl.download([url])
                print(f"Downloaded {name}.mp3 from {url}")
                return str(output_path)
            except: 
                return None
        
        with open(self.track_data_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            for song in self.songs:
                """
                Track hashes from the video api with 'revision_id' seem to exist in the 
                'tracks' object of each song.
                Sometimes video list has no track hashes (i.e. trackHashes: Null 
                (and maybe tarcks: Null also?)), in this case one default video 
                is used for all tracks. This is just the only video in the list 
                or the one that doesn't have feature: "alternative" or maybe just needs
                "feature": null,
                """
                if song['song_id'] in self.audio_video_scraped_song_ids_set and not start_from_start:
                    continue

                try:
                    tracks = self._safe_get(song, ['tracks'])
                    if not tracks:
                        print(f"No tracks found for {song['name']}")
                        continue

                    guitar_tracks = [track for track in tracks if track['hash'].startswith('guitar')]
                    if not guitar_tracks:
                        print(f"No guitar tracks found for {song['name']}")
                        continue

                    video_list_url = f"https://www.songsterr.com/api/video-points/{song['song_id']}/{song['revision_id']}/list"
                    video_dicts = get_json(video_list_url)

                    if not video_dicts:
                        print(f"No video data found for {song['name']} at {video_list_url}")
                        continue
                        
                    # Assume default video is the first one to not have `feature: "alternative"`
                    default_video, default_points = next(
                        ((video['videoId'], video['points']) for video in video_dicts if video['feature'] != "alternative"),
                        (None, None)  # Default values if no match is found
                    )

                    # Check case where there are no hashes
                    video_dict_hashes_exist = [
                        video['trackHashes'] for video in video_dicts 
                        if video.get('trackHashes') not in [None, []]
                    ]

                    for track in guitar_tracks:
                        # Determine the video ID and points
                        if video_dict_hashes_exist:
                            track_video_id, track_points = next(
                                ((video['videoId'], video['points']) 
                                    for video in video_dicts 
                                    if isinstance(video.get('trackHashes'), list) and track['hash'] in video['trackHashes']
                                ),
                                (default_video, default_points)
                            )
                        else:
                            track_video_id, track_points = default_video, default_points
                        if not track_video_id in video_ids:
                            success = download_youtube_audio(track_video_id, f"{song['name']}_{track['partId']}")
                            video_ids.add(track_video_id)
                            if not success:
                                continue
                    
                        # Save tab data
                        if not song['image']:
                            print(f"No image found for {song['name']}")
                            continue
                        tab_url = (
                            f"https://dqsljvtekg760.cloudfront.net/{song['song_id']}/"
                            f"{song['revision_id']}/{song['image']}/{track['partId']}.json"
                        )
                        tab = get_json(tab_url)
                        with open(os.path.join(self.tabs_dir, f"{song['name']}_{track['partId']}.json"), 'w') as file:
                            json.dump(tab, file, indent=4)
                            print(f"Downloaded {song['name']}_{track['partId']} tab data")

                        writer.writerow([
                            f"{song['name']}_{track['partId']}",
                            song['song_id'],
                            song['revision_id'],
                            track['hash'],
                            track_video_id,
                            track_points,
                            f"{song['name']}_{track['partId']}"
                        ])

                        
                except Exception as e:
                    print(f"Failed {song['name']}: {e}")
                    continue
            







