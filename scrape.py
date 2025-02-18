import requests
import csv
import aiohttp
import asyncio

from scrapers.get_song_urls import get_urls
from scrapers.get_song_meta_data import get_song_meta_data
from scrapers.get_model_data import get_model_data


async def main():
  url_file = 'data/songsterr-data/song_urls.txt'
  song_meta_data_file = 'data/songsterr-data/song_meta_data.csv'

  session = requests.Session()
  session.headers.setdefault('User-Agent', 'Mediapartners-Google*')

  # get_urls(url_file=url_file, session=session)

  # with open(url_file, newline='', encoding='utf-8') as file:
  #   song_urls = [row.strip() for row in file]  #

  # num_songs_to_scrape = 15000000
  # get_song_meta_data(song_urls=song_urls, song_meta_data_file=song_meta_data_file, session=session, num_songs=num_songs_to_scrape)

  with open(song_meta_data_file, newline='', encoding='utf-8') as csvfile:
    song_meta_data = list(csv.reader(csvfile))

  async with aiohttp.ClientSession() as session:
    await get_model_data(
      song_meta_data=song_meta_data,
      checkpoint_file='data/songsterr-data/checkpoint.csv',
      model_data_path_prefix='data/model_data/audio_tabs',
      session=session,
      batch_size=200,
    )


if __name__ == '__main__':
  asyncio.run(main())
