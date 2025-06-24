"""
Run NER model comparison.
"""
import argparse
from scripts.compare_models import NERModelComparator
from scripts.config import MODELS_DIR, LABELED_DATA_DIR, REPORTS_DIR
import os

def main():
    parser = argparse.ArgumentParser(description="Compare NER models for Amharic e-commerce data")
    parser.add_argument("--conll_path", default=os.path.join(LABELED_DATA_DIR, "labeled.conll"),
                        help="Path to labeled CoNLL file")
    parser.add_argument("--output_path", default=os.path.join(REPORTS_DIR, "model_comparison.csv"),
                        help="Path to save comparison CSV")
    args = parser.parse_args()

    models = [
        {"name": "XLM-RoBERTa", "model_name": "xlm-roberta-base", "output_dir": os.path.join(MODELS_DIR, "ner_xlmr")},
        {"name": "DistilBERT", "model_name": "distilbert-base-multilingual-cased", "output_dir": os.path.join(MODELS_DIR, "ner_distilbert")},
        {"name": "mBERT", "model_name": "bert-base-multilingual-cased", "output_dir": os.path.join(MODELS_DIR, "ner_mbert")}
    ]

    comparator = NERModelComparator(
        models=models,
        conll_path=args.conll_path,
        output_path=args.output_path
    )
    comparator.compare()

if __name__ == "__main__":
    main()
