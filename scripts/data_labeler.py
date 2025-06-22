"""
DataLabeler: Label Telegram messages for NER in CoNLL format
"""
import pandas as pd
import os
import re
import logging
from typing import List, Tuple
from scripts.config import RAW_DATA_DIR, LABELED_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("labeler.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataLabeler:
    def __init__(self, input_path: str, output_path: str, sample_size: int = 50):
        """
        Initialize the data labeler.

        Args:
            input_path (str): Path to input CSV (scraped.csv)
            output_path (str): Path to output CoNLL file (labeled.conll)
            sample_size (int): Number of messages to label
        """
        self.input_path = input_path
        self.output_path = output_path
        self.sample_size = sample_size
        self.valid_tags = [
            "O",
            "B-Product", "I-Product",
            "B-Price", "I-Price",
            "B-Location", "I-Location",
            "B-Delivery", "I-Delivery",
            "B-Contact", "I-Contact"
        ]
        # Amharic/English location names
        self.location_keywords = [
            "መገናኛ", "ቦሌ", "ሜክሲኮ", "መሰረት", "ደፋርሞል",
            "ሁለተኛፎቅ", "አለምነሽ", "ፕላዛ", "መድሐኔዓለም",
            "mexico", "bole"
        ]
        # Delivery keywords
        self.delivery_keywords = [
            "በሞተረኞች", "እናደርሳለን", "ያሉበት", "ከነፃ", "ዲሊቨሪ", "Free"
        ]
        # Promotional/greeting keywords to filter
        self.promo_keywords = [
            "በረፍት", "ሱቅ ላይ", "እንገኛለን", "ቅናሽ", "እንኳን", "ሱቃችን", "Eid"
        ]

    def load_data(self) -> pd.DataFrame:
        """
        Load and sample messages from scraped.csv with stratified sampling.

        Returns:
            pd.DataFrame: Sampled messages
        """
        try:
            df = pd.read_csv(self.input_path, encoding="utf-8")
            logger.info(f"Loaded {len(df)} messages from {self.input_path}")
            if len(df) < self.sample_size:
                logger.warning(f"Only {len(df)} messages available, sampling all")
                return df
            
            # Filter out promotional/greeting posts
            df = df[~df['text'].str.contains('|'.join(self.promo_keywords), na=False)]
            logger.info(f"After filtering promotional posts, {len(df)} messages remain")
            
            # Stratified sampling by channel
            channel_counts = df['channel'].value_counts()
            logger.info(f"Found {len(channel_counts)} unique channels: {channel_counts.to_dict()}")
            sample_per_channel = max(1, self.sample_size // len(channel_counts))
            sampled_dfs = []
            
            for channel in channel_counts.index:
                channel_df = df[df['channel'] == channel]
                n_samples = min(len(channel_df), sample_per_channel)
                sampled_dfs.append(channel_df.sample(n=n_samples, random_state=42))
            
            sampled_df = pd.concat(sampled_dfs)
            # Adjust to exact sample_size
            if len(sampled_df) < self.sample_size:
                remaining = df[~df.index.isin(sampled_df.index)]
                if not remaining.empty:
                    extra_samples = remaining.sample(
                        n=min(self.sample_size - len(sampled_df), len(remaining)),
                        random_state=42
                    )
                    sampled_df = pd.concat([sampled_df, extra_samples])
            
            return sampled_df.sample(n=min(self.sample_size, len(sampled_df)), random_state=42)
        except Exception as e:
            logger.error(f"Failed to load {self.input_path}: {e}")
            raise

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words, removing leading/trailing ellipses.

        Args:
            text (str): Input text

        Returns:
            List[str]: List of tokens
        """
        if not text or not isinstance(text, str):
            return []
        # Remove leading/trailing ellipses
        text = re.sub(r"^\.+|\.+$", "", text.strip())
        # Space-based tokenization
        return text.strip().split()

    def suggest_tags(self, token: str, prev_token: str = "", prev_tag: str = "", tokens: List[str] = None, index: int = 0) -> str:
        """
        Suggest NER tags based on patterns in scraped.csv.

        Args:
            token (str): Current token
            prev_token (str): Previous token
            prev_tag (str): Previous tag
            tokens (List[str]): Full token list for context
            index (int): Current token index

        Returns:
            str: Suggested tag
        """
        # Contact: Phone numbers (09 or 251) or Telegram handles (@)
        if token.startswith("@") or re.match(r"^(09|251|\+251)\d{8}$", token.replace("+", "")):
            return "B-Contact"
        
        # Price: Number (with/without commas) followed by ብር or after ዋጋ፦/PRICE
        if re.match(r"^\d{1,3}(,\d{3})?$", token.replace(",", "")) and prev_token in ["ዋጋ፦", "ዋጋ:", "PRICE"]:
            return "B-Price"
        if token == "ብር" and prev_tag == "B-Price":
            return "I-Price"
        if re.match(r"^\d{1,3}(,\d{3})?$", token.replace(",", "")) and index + 1 < len(tokens) and tokens[index + 1] == "ብር":
            return "B-Price"
        
        # Location: Amharic or English place names
        if token in self.location_keywords:
            return "B-Location"
        if prev_tag.startswith("B-Location") and (token in self.location_keywords or re.match(r"^[A-Za-z0-9]+$", token)):
            return "I-Location"
        
        # Delivery: Delivery phrases
        if token in self.delivery_keywords:
            return "B-Delivery"
        if prev_tag == "B-Delivery" and token in self.delivery_keywords:
            return "I-Delivery"
        
        # Product: Sequence at start, before ዋጋ፦/PRICE, or after ellipses
        if index == 0 or prev_token in ["...", "…"] or (tokens and index < len(tokens) - 1 and tokens[index + 1] in ["ዋጋ፦", "ዋጋ:", "PRICE"]):
            return "B-Product"
        if prev_tag.startswith("B-Product") and tokens and index + 1 < len(tokens) and tokens[index + 1] not in ["ዋጋ፦", "ዋጋ:", "PRICE", "ብር"]:
            return "I-Product"
        
        return "O"

    def label_message(self, message_id: str, channel: str, text: str) -> List[Tuple[str, str]]:
        """
        Label tokens in a message with user input.

        Args:
            message_id (str): Message ID
            channel (str): Channel name
            text (str): Message text

        Returns:
            List[Tuple[str, str]]: List of (token, tag) pairs
        """
        tokens = self.tokenize(text)
        if not tokens:
            return []

        labeled_tokens = []
        logger.info(f"\nLabeling message {message_id} from {channel}:")
        print(f"\nMessage: {text}")
        print(f"Tokens: {tokens}")
        print(f"Valid tags: {', '.join(self.valid_tags)}")

        prev_token = ""
        prev_tag = ""
        for i, token in enumerate(tokens):
            suggested_tag = self.suggest_tags(token, prev_token, prev_tag, tokens, i)
            print(f"\nToken {i+1}: {token}")
            print(f"Suggested tag: {suggested_tag}")
            tag = input("Enter tag (press Enter to accept suggested tag, or type new tag): ").strip()
            tag = suggested_tag if tag == "" else tag

            while tag not in self.valid_tags:
                print(f"Invalid tag. Valid tags: {', '.join(self.valid_tags)}")
                tag = input("Enter tag: ").strip() or suggested_tag

            labeled_tokens.append((token, tag))
            prev_token = token
            prev_tag = tag

        return labeled_tokens

    def save_conll(self, labeled_data: List[Tuple[str, str, str, str]]):
        """
        Save labeled data to CoNLL format.

        Args:
            labeled_data: List of (message_id, channel, text, [(token, tag), ...])
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for message_id, channel, text, tokens_tags in labeled_data:
                if not tokens_tags:
                    continue
                f.write(f"# message_id: {message_id}\n")
                f.write(f"# channel: {channel}\n")
                f.write(f"# text: {text}\n")
                for token, tag in tokens_tags:
                    f.write(f"{token} {tag}\n")
                f.write("\n")
        logger.info(f"Saved labeled data to {self.output_path}")

    def run(self):
        """
        Run the labeling process.
        """
        logger.info("Starting data labeling")
        df = self.load_data()
        labeled_data = []

        for _, row in df.iterrows():
            message_id = str(row["message_id"])
            channel = row["channel"]
            text = row["text"]
            if pd.isna(text) or not text.strip():
                logger.warning(f"Skipping empty text for message_id {message_id}")
                continue
            tokens_tags = self.label_message(message_id, channel, text)
            if tokens_tags:
                labeled_data.append((message_id, channel, text, tokens_tags))

        self.save_conll(labeled_data)
        logger.info("Labeling completed")

if __name__ == "__main__":
    labeler = DataLabeler(
        input_path=os.path.join(RAW_DATA_DIR, "scraped.csv"),
        output_path=os.path.join(LABELED_DATA_DIR, "labeled.conll"),
        sample_size=50
    )
    labeler.run()
