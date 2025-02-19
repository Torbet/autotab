import requests
import csv
import aiohttp
import asyncio

from scrapers.get_model_data import get_model_data


async def main():
  session = requests.Session()
  session.headers.setdefault('User-Agent', 'Mediapartners-Google*')

  song_meta_data_file = 'data/songsterr-data/song_meta_data.csv'
  with open(song_meta_data_file, newline='', encoding='utf-8') as csvfile:
    song_meta_data = list(csv.reader(csvfile))

  async with aiohttp.ClientSession() as session:
    await get_model_data(
      song_meta_data=song_meta_data,
      checkpoint_file='data/songsterr-data/checkpoint.csv',
      model_data_path_prefix='data/model_data/audio_tabs',
      session=session,
      batch_size=200,
      num_batches=200,
    )


if __name__ == '__main__':
  asyncio.run(main())
