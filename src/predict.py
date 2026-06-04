"""
predict.py
==========
Inference module for MalScan-ML.

Loads a trained model and predicts on new PE files.
Used internally by scan.py.

Usage:
    from predict import MalwarePredictor

    predictor = MalwarePredictor()
    result = predictor.predict("file.exe")
    print(result)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import joblib
import pandas as pd

from feature_extractor import PEFeatureExtractor
from utils import get_feature_columns, create_sample_dataset

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


class MalwarePredictor:
    """
    Loads trained model and predicts malware probability for PE files.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to .pkl model file.
                        Defaults to models/best_model.pkl
        """
        self.model_path = Path(model_path) if model_path else MODELS_DIR / "best_model.pkl"
        self.extractor = PEFeatureExtractor()
        self.model = None
        self.model_name = "unknown"
        self.feature_names = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Run 'python src/train.py --data data/processed/features.csv' first."
            )
        self.model = joblib.load(self.model_path)

        # Load model name if available
        name_path = MODELS_DIR / "best_model_name.pkl"
        if name_path.exists():
            self.model_name = joblib.load(name_path)

        # Infer expected feature names from a sample record
        sample_df = create_sample_dataset(n_samples=2)
        self.feature_names = get_feature_columns(sample_df)
        logger.debug(f"Model loaded: {self.model_name} ({len(self.feature_names)} features)")

    def predict(self, filepath: str) -> dict:
        """
        Predict malware probability for a single file.

        Args:
            filepath: Path to PE file

        Returns:
            dict with keys: verdict, confidence, label, features, error
        """
        filepath = Path(filepath)

        # Extract features
        features = self.extractor.extract(str(filepath))
        if features is None:
            return {
                "verdict": "ERROR",
                "confidence": 0.0,
                "label": -1,
                "features": {},
                "error": "Could not parse PE file.",
                "filepath": str(filepath),
            }

        # Build feature vector in correct order
        try:
            feature_vector = np.array([
                features.get(name, 0.0) for name in self.feature_names
            ]).reshape(1, -1)
        except Exception as e:
            return {
                "verdict": "ERROR",
                "confidence": 0.0,
                "label": -1,
                "features": features,
                "error": f"Feature vector error: {e}",
                "filepath": str(filepath),
            }

        # Predict
        try:
            proba = self.model.predict_proba(feature_vector)[0]
            label = int(proba[1] >= 0.5)
            confidence = float(proba[1] if label == 1 else proba[0])
        except Exception as e:
            return {
                "verdict": "ERROR",
                "confidence": 0.0,
                "label": -1,
                "features": features,
                "error": f"Prediction error: {e}",
                "filepath": str(filepath),
            }

        verdict = "MALWARE" if label == 1 else "BENIGN"

        # Get top risk indicators (features with highest values relative to malware pattern)
        risk_indicators = self._get_risk_indicators(features)

        return {
            "verdict": verdict,
            "confidence": round(confidence * 100, 2),
            "label": label,
            "features": features,
            "risk_indicators": risk_indicators,
            "model_name": self.model_name,
            "filepath": str(filepath),
            "error": None,
        }

    def _get_risk_indicators(self, features: dict) -> list[str]:
        """
        Generate human-readable risk indicator messages based on feature values.
        """
        indicators = []

        if features.get("file_entropy", 0) > 7.0:
            indicators.append(f"High file entropy ({features['file_entropy']:.2f}) — likely packed/encrypted")

        if features.get("mean_entropy", 0) > 6.5:
            indicators.append(f"High section entropy (mean={features['mean_entropy']:.2f})")

        if features.get("num_suspicious_imports", 0) > 3:
            indicators.append(f"Many suspicious API imports ({features['num_suspicious_imports']})")

        if features.get("has_createremotethread", 0):
            indicators.append("CreateRemoteThread import — process injection capability")

        if features.get("has_writeprocessmemory", 0):
            indicators.append("WriteProcessMemory import — memory manipulation capability")

        if features.get("has_isdebuggerpresent", 0):
            indicators.append("IsDebuggerPresent — anti-debugging technique")

        if features.get("has_urldownloadtofile", 0):
            indicators.append("URLDownloadToFile — remote file download capability")

        if features.get("suspicious_section_names", 0) > 0:
            indicators.append("Packer-associated section names detected (UPX, ASPack, etc.)")

        if features.get("ep_in_unusual_section", 0):
            indicators.append("Entry point in non-standard section")

        if features.get("wx_sections", 0) > 0:
            indicators.append(f"Writable+Executable sections ({features['wx_sections']}) — W^X violation")

        if features.get("compile_timestamp", 0) == 0:
            indicators.append("Zero compile timestamp — metadata stripped")

        if features.get("has_debug_info", 1) == 0:
            indicators.append("No debug information — stripped binary")

        if features.get("num_b64_strings", 0) > 5:
            indicators.append(f"Multiple base64 strings ({features['num_b64_strings']}) — possible obfuscation")

        if features.get("imports_ws2_32", 0) or features.get("imports_wininet", 0):
            indicators.append("Network library imports (Winsock/WinINet)")

        if not indicators:
            indicators.append("No strong risk indicators detected")

        return indicators[:5]  # Top 5 most relevant
