import re
import pickle
import logging
from pathlib import Path
from app.decorators import timer, log_prediction

logger = logging.getLogger(__name__)

class FakeNewsDetector:
    def __init__(self, model_path: str, vectorizer_path: str) -> None:
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.model = self._load_pickle(self.model_path)
        self.vectorizer = self._load_pickle(self.vectorizer_path)
        logger.info("[DETECTOR] Modèle chargé depuis '%s'", model_path)

    @staticmethod
    def _load_pickle(path: Path) -> object:
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def _clean(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)  
        text = re.sub(r"[^a-z\s]", "", text)                 
        text = re.sub(r"\s+", " ", text).strip()             
        return text

    @timer
    @log_prediction
    def predict(self, text: str) -> dict:
  
        cleaned = self._clean(text)

        if not cleaned:
            return {"label": "UNKNOWN", "confidence": 0.0, "input_preview": text[:80]}

        vec = self.vectorizer.transform([cleaned])
        pred = self.model.predict(vec)[0]

        confidence = None
        if hasattr(self.model, "decision_function"):
            score = self.model.decision_function(vec)[0]
            confidence = round(1 / (1 + pow(2.718, -score)), 4)

        return {
            "label": "REAL" if pred == 1 else "FAKE",
            "confidence": confidence,
            "input_preview": text[:80],
        }

    def __repr__(self) -> str:
        return (
            f"FakeNewsDetector("
            f"model='{self.model_path.name}', "
            f"vectorizer='{self.vectorizer_path.name}')"
        )