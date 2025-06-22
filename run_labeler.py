"""
Run the DataLabeler to create NER dataset in CoNLL format.
"""
import logging
import argparse
from scripts.data_labeler import DataLabeler
from scripts.config import RAW_DATA_DIR, LABELED_DATA_DIR
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("labeler.log", encoding="utf-8", mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run NER data labeler")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of messages to label")
    args = parser.parse_args()

    labeler = DataLabeler(
        input_path=os.path.join(RAW_DATA_DIR, "scraped.csv"),
        output_path=os.path.join(LABELED_DATA_DIR, "labeled.conll"),
        sample_size=args.sample_size
    )
    labeler.run()

if __name__ == "__main__":
    main()
