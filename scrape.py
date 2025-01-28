from scrapers.songsterr_scraper import SongsterrScraper

def main():
    scraper = SongsterrScraper()
    scraper.get_urls()
    scraper.get_song_data()
    scraper.get_audio_and_tab()


if __name__ == "__main__":
    main()