import os
import logging
import pandas as pd
import ast
from datetime import datetime
from scripts.config import LABELED_DATA_DIR, REPORTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("vendor_analytics.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VendorAnalytics:
    def __init__(
        self,
        input_path: str = os.path.join(LABELED_DATA_DIR, "scraped_with_ner.csv"),
        output_path: str = os.path.join(REPORTS_DIR, "vendor_scorecard.csv")
    ):
        """
        Initialize vendor analytics engine.

        Args:
            input_path: Path to CSV with scraped posts and NER entities
            output_path: Path to save vendor scorecard CSV
        """
        self.input_path = input_path
        self.output_path = output_path
        self.df = None

    def load_data(self):
        """
        Load CSV data with NER entities.
        """
        try:
            self.df = pd.read_csv(self.input_path)
            # Convert stringified entities to lists
            self.df['entities'] = self.df['entities'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            logger.info(f"Loaded {len(self.df)} posts from {self.input_path}")
        except FileNotFoundError:
            logger.error(f"Input file {self.input_path} not found")
            raise

    def calculate_metrics(self) -> pd.DataFrame:
        """
        Calculate vendor metrics and Lending Score.

        Returns:
            DataFrame with vendor metrics
        """
        if self.df is None:
            self.load_data()

        results = []
        for vendor in self.df['channel'].unique():
            logger.info(f"Processing vendor: {vendor}")
            vendor_df = self.df[self.df['channel'] == vendor]

            # Posting Frequency (posts per week)
            timestamps = pd.to_datetime(vendor_df['timestamp'])
            if timestamps.empty:
                continue
            time_span_days = (timestamps.max() - timestamps.min()).days
            time_span_weeks = max(time_span_days / 7, 1)
            posts_per_week = len(vendor_df) / time_span_weeks

            # Average Views per Post
            avg_views = vendor_df['views'].mean()

            # Top-Performing Post
            top_post = vendor_df.loc[vendor_df['views'].idxmax()]
            top_product = "N/A"
            top_price = "N/A"
            for entity in top_post['entities']:
                if entity['entity'] == "B-Product":
                    top_product = entity['text']
                if entity['entity'] == "B-Price":
                    top_price = entity['text']

            # Average Price Point
            prices = []
            for entities in vendor_df['entities']:
                for entity in entities:
                    if entity['entity'] == "B-Price":
                        try:
                            price_str = "".join(c for c in entity['text'] if c.isdigit() or c == ".")
                            price = float(price_str)
                            prices.append(price)
                        except ValueError:
                            continue
            avg_price = sum(prices) / len(prices) if prices else 0

            # Lending Score
            lending_score = (avg_views * 0.4) + (posts_per_week * 0.3) + (avg_price * 0.3 / 1000)

            results.append({
                "Vendor": vendor,
                "Avg. Views/Post": round(avg_views, 2),
                "Posts/Week": round(posts_per_week, 2),
                "Avg. Price (ETB)": round(avg_price, 2),
                "Lending Score": round(lending_score, 2),
                "Top Post Product": top_product,
                "Top Post Price": top_price
            })

        df = pd.DataFrame(results)
        logger.info(f"Vendor scorecard:\n{df}")
        df.to_csv(self.output_path, index=False)
        logger.info(f"Saved vendor scorecard to {self.output_path}")
        return df

if __name__ == "__main__":
    analytics = VendorAnalytics()
    analytics.calculate_metrics()

