"""
Fine-tune XLM-RoBERTa for NER on labeled.conll.
"""
import os
import json
import logging
from typing import List, Dict
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from seqeval.metrics import precision_score, recall_score, f1_score
from scripts.config import LABELED_DATA_DIR, MODELS_DIR, REPORTS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fine_tune_ner.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NERFineTuner:
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        conll_path: str = os.path.join(LABELED_DATA_DIR, "labeled.conll"),
        output_dir: str = os.path.join(MODELS_DIR, "ner_xlmr"),
        report_path: str = os.path.join(REPORTS_DIR, "ner_metrics.json"),
        max_length: int = 128,
        batch_size: int = 4,  # Reduced for CPU
        epochs: int = 3
    ):
        """
        Initialize NER fine-tuner.

        Args:
            model_name: Pretrained model name (e.g., xlm-roberta-base)
            conll_path: Path to labeled.conll
            output_dir: Directory to save model
            report_path: Path to save metrics
            max_length: Max token length
            batch_size: Training batch size
            epochs: Number of training epochs
        """
        self.model_name = model_name
        self.conll_path = conll_path
        self.output_dir = output_dir
        self.report_path = report_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_list = [
            "O",
            "B-Product", "I-Product",
            "B-Price", "I-Price",
            "B-Location", "I-Location",
            "B-Contact", "I-Contact",
            "B-Delivery", "I-Delivery"
        ]
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}

    def load_conll(self) -> List[Dict]:
        """
        Load CoNLL data into a list of dictionaries.

        Returns:
            List of dicts with tokens, ner_tags, and metadata
        """
        data = []
        current_message = {"tokens": [], "ner_tags": [], "metadata": {}}
        with open(self.conll_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    if "message_id" in line:
                        current_message["metadata"]["message_id"] = line.split(": ")[1]
                    elif "channel" in line:
                        current_message["metadata"]["channel"] = line.split(": ")[1]
                    elif "text" in line:
                        current_message["metadata"]["text"] = line.split(": ")[1]
                elif line:
                    token, tag = line.split()
                    current_message["tokens"].append(token)
                    current_message["ner_tags"].append(tag)
                elif current_message["tokens"]:
                    data.append(current_message)
                    current_message = {"tokens": [], "ner_tags": [], "metadata": {}}
        if current_message["tokens"]:
            data.append(current_message)
        logger.info(f"Loaded {len(data)} messages from {self.conll_path}")
        return data

    def align_tokens_labels(self, example: Dict) -> Dict:
        """
        Align tokens with subword tokenization and convert tags to IDs for a single example.

        Args:
            example: Dict with tokens and ner_tags

        Returns:
            Dictionary with tokenized inputs and aligned labels
        """
        tokenized_inputs = self.tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None  # Return Python lists
        )

        word_ids = tokenized_inputs.word_ids()
        label_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(self.label2id[example["ner_tags"][word_idx]])
            else:
                label_ids.append(self.label2id[example["ner_tags"][word_idx]])
            prev_word_idx = word_idx

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": label_ids
        }

    def prepare_dataset(self, data: List[Dict]) -> Dict:
        """
        Prepare dataset for training.

        Args:
            data: List of dicts from load_conll

        Returns:
            Train and validation datasets
        """
        dataset = Dataset.from_list(data)
        train_data, val_data = train_test_split(
            data, test_size=0.2, random_state=42
        )
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)

        tokenized_train = train_dataset.map(
            self.align_tokens_labels,
            batched=False,
            remove_columns=["tokens", "ner_tags", "metadata"]
        )
        tokenized_val = val_dataset.map(
            self.align_tokens_labels,
            batched=False,
            remove_columns=["tokens", "ner_tags", "metadata"]
        )

        logger.info(f"Train dataset sample: {tokenized_train[0]}")
        logger.info(f"Validation dataset sample: {tokenized_val[0]}")
        logger.info(f"Prepared {len(train_dataset)} train and {len(val_dataset)} validation examples")
        return {"train": tokenized_train, "val": tokenized_val}


    def compute_metrics(self, eval_pred) -> Dict:
        """
        Compute precision, recall, F1 using seqeval.

        Args:
        eval_pred: Tuple of predictions and labels

        Returns:
        Dict with metrics
        """
        predictions, labels = eval_pred
        predictions = torch.argmax(torch.tensor(predictions), dim=-1).numpy()
        true_labels = labels

        pred_tags = []
        true_tags = []
        for pred, true in zip(predictions, true_labels):
            pred_seq = []
            true_seq = []
            for p, t in zip(pred, true):
                if t != -100:
                    pred_seq.append(self.id2label[p])
                    true_seq.append(self.id2label[t])
            pred_tags.append(pred_seq)
            true_tags.append(true_seq)

        metrics = {
            "precision": precision_score(true_tags, pred_tags),
            "recall": recall_score(true_tags, pred_tags),
            "f1": f1_score(true_tags, pred_tags)
        }
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics


    def run(self):
        """
        Run fine-tuning process.
        """
        logger.info("Starting NER fine-tuning")
        data = self.load_conll()
        datasets = self.prepare_dataset(data)

        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=2,  # Simulate batch size 8
            warmup_steps=10,  # Reduced for small dataset
            weight_decay=0.01,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=5,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            max_steps=30  # 40 examples / effective batch_size 8 (4*2) ≈ 5 steps/epoch * 3 epochs
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.max_length
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["val"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Saved model to {self.output_dir}")

        metrics = trainer.evaluate()
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved metrics to {self.report_path}")

if __name__ == "__main__":
    fine_tuner = NERFineTuner()
    fine_tuner.run()
