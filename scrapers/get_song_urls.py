import os
from typing import List
import requests
from bs4 import BeautifulSoup

def get_urls(url_file: str, session: requests.Session) -> List[str]:
    if os.path.exists(url_file):
        with open(url_file, 'r', encoding='utf-8') as f:
            print("Found urls at: ", url_file)
            return [line.strip() for line in f if line.strip()]
    
    try:
        response = session.get('https://www.songsterr.com/sitemap-tabs.xml')
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to scrape main sitemap: {e}")
    
    soup = BeautifulSoup(response.content, 'lxml')
    page_urls = [loc.text for sitemap in soup.find_all('sitemap') if (loc := sitemap.find('loc'))]
    
    song_urls = []
    for page_url in page_urls:
        try:
            response = session.get(page_url)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to scrape {page_url}: {e}")
            continue
        soup = BeautifulSoup(response.content, 'lxml')
        song_urls.extend([loc.text for url in soup.find_all('url') if (loc := url.find('loc'))])
        print(f"Found {len(song_urls)} song URLs so far.")
    
    with open(url_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(song_urls))
    print(f"{len(song_urls)} song URLs saved to '{url_file}'.")
    
    return song_urls