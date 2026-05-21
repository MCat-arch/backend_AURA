"""
LSTMService: TFLite inference wrapper for panic detection LSTM model.

Model: lstm_panic_ft.tflite
Input shape: (batch, 7, 9)  — 7 timesteps, 9 features
Output: single float (sigmoid probability of panic, 0.0–1.0)

9 Features (ordered):
    bpm, mean_rr, sdnn, rmssd, pnn50, cv_rr, min_rr, max_rr, nn50

Normalization: Z-score using WESAD mean/std from model_meta_panic.json
"""

import json
import logging
import numpy as np

logger = logging.getLogger("LSTMService")

import tensorflow as tf


class LSTMService:
    """TFLite inference wrapper for the panic detection LSTM model."""

    def __init__(self, model_path: str, meta_path: str):
        """
        Initialize LSTMService.

        Args:
            model_path: Path to best_ft.keras
            meta_path: Path to model_meta_panic.json
        """
        # Load Keras model
        self.model = tf.keras.models.load_model(model_path)

        # Load metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.feature_names = meta["feature_names"]
        self.norm_mean = np.array(meta["norm_mean"], dtype=np.float32)
        self.norm_std = np.array(meta["norm_std"], dtype=np.float32)
        self.threshold = meta["threshold"]  # 0.55
        self.seq_len = meta["seq_len"]  # 7
        self.n_features = meta["n_features"]  # 9

        logger.info(
            f"LSTMService initialized | model={model_path} | "
            f"seq_len={self.seq_len} | n_features={self.n_features} | "
            f"threshold={self.threshold}"
        )

    def extract_features(self, bpm: int, hrv_data: dict) -> list[float]:
        """
        Extract 9 features from incoming data point.

        Mirrors the notebook's `hrv_to_9feat()` function.
        Derived features (cv_rr, min_rr, max_rr) are estimated from meanRR and sdnn.

        Args:
            bpm: Heart rate BPM value
            hrv_data: Dict with keys: meanRR, sdnn, rmssd, pnn50, nn50
                      (from HRV60s)

        Returns:
            List of 9 float features in order:
            [bpm, mean_rr, sdnn, rmssd, pnn50, cv_rr, min_rr, max_rr, nn50]
        """
        mean_rr = hrv_data.get("meanRR") or 0.0
        sdnn = hrv_data.get("sdnn") or 0.0
        rmssd = hrv_data.get("rmssd") or 0.0
        pnn50 = hrv_data.get("pnn50") or 0.0
        nn50 = hrv_data.get("nn50") or 0.0

        # BPM: use from HRV's meanRR if available, otherwise from input bpm
        feat_bpm = 60000.0 / mean_rr if mean_rr > 0 else float(bpm)

        # Derived features (matching notebook's hrv_to_9feat)
        cv_rr = (sdnn / mean_rr * 100) if mean_rr > 0 else 0.0
        min_rr = max(300.0, mean_rr - 2.5 * sdnn)
        max_rr = min(2000.0, mean_rr + 2.5 * sdnn)

        return [feat_bpm, mean_rr, sdnn, rmssd, pnn50, cv_rr, min_rr, max_rr, nn50]

    def predict(self, feature_window: list[list[float]]) -> float:
        """
        Run LSTM inference on a window of feature vectors.

        Args:
            feature_window: List of 7 feature vectors (each with 9 floats)

        Returns:
            Panic probability (float 0.0–1.0)
        """
        if len(feature_window) != self.seq_len:
            raise ValueError(
                f"Expected {self.seq_len} data points, got {len(feature_window)}"
            )

        # Convert to numpy array
        arr = np.array(feature_window, dtype=np.float32)

        # Normalize with WESAD statistics
        arr_norm = (arr - self.norm_mean) / self.norm_std

        # Reshape to (1, 7, 9) — batch dimension
        input_data = arr_norm[np.newaxis]

        # Run inference
        output = self.model.predict(input_data, verbose=0)

        prob = float(output[0, 0])
        logger.debug(f"LSTM inference result: p_panic={prob:.4f}")

        return prob
