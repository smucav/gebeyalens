import argparse
from scripts.vendor_analytics import VendorAnalytics
from scripts.config import LABELED_DATA_DIR, REPORTS_DIR
import os

def main():
    parser = argparse.ArgumentParser(description="Generate vendor scorecard for micro-lending")
    parser.add_argument("--input_path", default=os.path.join(LABELED_DATA_DIR, "scraped_with_ner.csv"),
                        help="Path to CSV with scraped posts and NER entities")
    parser.add_argument("--output_path", default=os.path.join(REPORTS_DIR, "vendor_scorecard.csv"),
                        help="Path to save vendor scorecard CSV")
    args = parser.parse_args()

    analytics = VendorAnalytics(
        input_path=args.input_path,
        output_path=args.output_path
    )
    analytics.calculate_metrics()

if __name__ == "__main__":
    main()
