import os
import logging
import pandas as pd
from transformers import pipeline
from scripts.config import MODELS_DIR, LABELED_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ner_predictions.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_ner_predictions(
    input_path: str = os.path.join(LABELED_DATA_DIR, "scraped.csv"),
    output_path: str = os.path.join(LABELED_DATA_DIR, "scraped_with_ner.csv"),
    model_path: str = os.path.join(MODELS_DIR, "ner_xlmr")
):
    """
    Generate NER predictions for scraped CSV data using XLM-RoBERTa.

    Args:
        input_path: Path to scraped CSV data
        output_path: Path to save CSV with NER entities
        model_path: Path to fine-tuned XLM-RoBERTa model
    """
    logger.info(f"Loading NER model from {model_path}")
    ner_pipeline = pipeline("ner", model=model_path, tokenizer=model_path, aggregation_strategy="simple")

    logger.info(f"Loading scraped data from {input_path}")
    df = pd.read_csv(input_path)

    # Initialize entities column
    df['entities'] = None

    # Apply NER to each post
    for idx, row in df.iterrows():
        text = str(row['text'])
        if text and text != 'nan':
            logger.info(f"Processing post from {row['channel']}: {text[:50]}...")
            predictions = ner_pipeline(text)
            df.at[idx, 'entities'] = [
                {"entity": pred["entity_group"], "text": pred["word"]}
                for pred in predictions
            ]
        else:
            df.at[idx, 'entities'] = []

    logger.info(f"Saving output to {output_path}")
    df.to_csv(output_pathoutput, index=False)

if __name__ == "__main__":
    generate_ner_predictions()
