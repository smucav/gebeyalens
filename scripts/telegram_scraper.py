"""
TelegramScraper: Fetch messages and product images from Amharic e-commerce Telegram channels
"""
import asyncio
import os
import re
import logging
import pandas as pd
from datetime import datetime
from typing import List, Optional

from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("scraper.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TelegramScraper:
    def __init__(self, api_id: str, api_hash: str, channels: List[str], output_dir: str):
        """
        Initialize the Telegram client and create required directories.
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.channels = channels
        self.output_dir = output_dir
        self.media_dir = os.path.join(output_dir, "photos")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.media_dir, exist_ok=True)
        self.client = TelegramClient("scraper_session", api_id, api_hash)

    async def connect(self):
        """
        Start the Telegram client session.
        """
        await self.client.start()
        logger.info("Connected to Telegram")

    def preprocess_text(self, text: Optional[str]) -> str:
        """
        Clean and normalize Amharic + mixed-language text.
        """
        if not text or not isinstance(text, str):
            return ""
        text = re.sub(r'[^\u1200-\u137F\sA-Za-z\d.,!?@]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    async def download_image(self, message, channel: str) -> Optional[str]:
        """
        Download product image if present, return local path.
        """
        try:
            if message.media and hasattr(message.media, 'photo'):
                filename = f"{channel}_{message.id}.jpg"
                path = os.path.join(self.media_dir, filename)
                await self.client.download_media(message.media, path)
                return path
        except Exception as e:
            logger.warning(f"⚠️ Failed to download image from {channel} msg {message.id}: {e}")
        return None

    async def scrape_channel(self, channel: str, limit: int = 100) -> List[dict]:
        """
        Scrape messages and images from a single Telegram channel.
        """
        logger.info(f"Scraping channel: {channel}")
        messages = []

        try:
            entity = await self.client.get_entity(channel)
            history = await self.client(GetHistoryRequest(
                peer=entity,
                limit=limit,
                offset_date=None,
                offset_id=0,
                max_id=0,
                min_id=0,
                add_offset=0,
                hash=0
            ))

            for msg in history.messages:
                clean_text = self.preprocess_text(msg.message)
                if not clean_text:
                    continue

                image_path = await self.download_image(msg, channel)

                messages.append({
                    "channel": channel,
                    "message_id": msg.id,
                    "text": clean_text,
                    "timestamp": msg.date.isoformat(),
                    "views": msg.views or 0,
                    "media_path": image_path or ""
                })

        except Exception as e:
            logger.error(f"Failed to scrape {channel}: {e}")

        return messages

    async def scrape_all(self, limit_per_channel: int = 100):
        """
        Scrape all configured channels and export to CSV.
        """
        await self.connect()
        all_messages = []

        for channel in self.channels:
            msgs = await self.scrape_channel(channel, limit_per_channel)
            all_messages.extend(msgs)

        # Save to CSV
        output_path = os.path.join(self.output_dir, "scraped.csv")
        df = pd.DataFrame(all_messages)
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"✅ Saved {len(df)} messages to {output_path}")

        await self.client.disconnect()
