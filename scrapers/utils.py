from typing import List, Any, Set
import csv
import os


def safe_get(data: dict, keys: List[str], default: Any = None) -> Any:
  """Safely get a nested value from a dictionary."""
  try:
    for key in keys:
      data = data[key]
    return data
  except Exception:
    return default


def load_scraped_song_ids(song_data_file: str) -> Set[str]:
  """
  Loads scraped song IDs from a CSV file.
  Returns an empty set if the file doesn't exist.
  """
  if not os.path.exists(song_data_file):
    return set()

  with open(song_data_file, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    # Skip header row
    return {row[0] for row in reader if row}


def get_json(url, session):
  try:
    response = session.get(url)
    response.raise_for_status()
  except Exception as e:
    print(f'Failed to get JSON from {url}: {e}')
    return None
  return response.json()
