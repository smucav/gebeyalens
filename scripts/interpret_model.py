"""
Interpret NER model predictions using SHAP.
"""
import os
import logging
import shap
from transformers import pipeline
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import MODELS_DIR, REPORTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("interpret_model.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NERInterpreter:
    def __init__(
        self,
        model_path: str = os.path.join(MODELS_DIR, "ner_xlmr"),
        output_path: str = os.path.join(REPORTS_DIR, "shap_plot.html")
    ):
        """
        Initialize NER interpreter with SHAP.

        Args:
            model_path: Path to fine-tuned model
            output_path: Path to save SHAP plot
        """
        self.model_path = model_path
        self.output_path = output_path
        self.ner_pipeline = pipeline("ner", model=model_path, tokenizer=model_path)

    def explain(self, text: str):
        """
        Generate SHAP explanation for a single text.

        Args:
            text: Input text for NER
        """
        logger.info(f"Generating SHAP explanation for text: {text}")
        explainer = shap.Explainer(self.ner_pipeline)
        shap_values = explainer([text])
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(shap.plots.text(shap_values[0], display=False).html())
        logger.info(f"Saved SHAP plot to {self.output_path}")

if __name__ == "__main__":
    interpreter = NERInterpreter()
    sample_text = "ጫማ በ 2000 ብር በአዲስ አበባ"
    interpreter.explain(sample_text)
