"""
Run NER fine-tuning script.
"""
import argparse
import logging
from scripts.fine_tune_ner import NERFineTuner
from scripts.config import LABELED_DATA_DIR, MODELS_DIR, REPORTS_DIR
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fine_tune_ner.log", encoding="utf-8", mode="a"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune NER model")
    parser.add_argument("--model-name", type=str, default="xlm-roberta-base", help="Pretrained model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-length", type=int, default=128, help="Max token length")
    args = parser.parse_args()

    fine_tuner = NERFineTuner(
        model_name=args.model_name,
        conll_path=os.path.join(LABELED_DATA_DIR, "labeled.conll"),
        output_dir=os.path.join(MODELS_DIR, "ner_xlmr"),
        report_path=os.path.join(REPORTS_DIR, "ner_metrics.json"),
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    fine_tuner.run()

if __name__ == "__main__":
    main()
