"""
Global configuration for EthioMart Amharic NER Project
"""
import os


# List of Telegram channels to scrape
TELEGRAM_CHANNELS = [
    "ZemenExpress",
    "nevacomputer",
    "meneshayeofficial"
    "ethio_brand_collection",
    "Leyueqa",
    "sinayelj",
    "Shewabrand",
    "helloomarketethiopia",
    "modernshoppingcenter",
    "qnashcom",
    "Fashiontera",
    "kuruwear",
    "gebeyaadama",
    "MerttEka",
    "forfreemarket",
    "classybrands",
    "marakibrand",
    "aradabrand2",
    "marakisat2",
    "belaclassic",
    "AwasMart",
]

# Telegram API credentials (get from https://my.telegram.org)
TELEGRAM_API_ID = os.getenv("APP_ID")
TELEGRAM_API_HASH = os.getenv("APP_HASH")

# Paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
LABELED_DATA_DIR = "data/labeled"
