"""
Run the TelegramScraper to fetch messages from e-commerce channels.
"""
import asyncio
import logging
from scripts.telegram_scraper import TelegramScraper
from scripts.config import TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_CHANNELS, RAW_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log', encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Validate configuration
    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
        logger.error("TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in config.py or environment variables")
        raise ValueError("Missing Telegram API credentials")

    # Initialize scraper
    scraper = TelegramScraper(
        api_id=TELEGRAM_API_ID,
        api_hash=TELEGRAM_API_HASH,
        channels=TELEGRAM_CHANNELS,
        output_dir=RAW_DATA_DIR
    )

    # Run scraper
    logger.info("Starting Telegram scraper")
    asyncio.run(scraper.scrape_all(limit_per_channel=50))
    logger.info("Scraping completed")

if __name__ == "__main__":
    main()
