# -*- coding: utf-8 -*-
"""
Pattern Confidence Engine
=========================
Mines Oracle-labeled bars to build a database of "winning setups".
At inference time, computes cosine similarity between the current bar's
features and the database of past winning patterns to produce a
pattern-based confidence score (0-1).

How it works:
  1. Training: Extract features from all LONG and SHORT oracle-labeled bars
               (these are ground-truth profitable setups)
  2. Inference: For a live bar, find the K most similar past winning setups
                using KNN with inverse-distance weighting
  3. Output:    Weighted average expected return → normalized to 0-1 confidence

This confidence score is used by the backtester for dynamic position sizing:
  high confidence → larger position size
  low confidence  → smaller position size
"""

import logging
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Features used for pattern matching — subset of full feature set,
# chosen because they best describe entry quality for scalping
PATTERN_FEATURES = [
    "rsi_14",
    "macd_hist",
    "bb_position",
    "adx",
    "volume_ratio_5",
    "candle_directional_score",
    "micro_position_6",
    "atr_ratio",
    "ema_9_21_spread",
    "roc_3",
    "roc_6",
    "body_pct_range",
    "buy_pressure",
]

PATTERN_DB_PATH = "models/pattern_confidence_db.pkl"


class PatternConfidenceEngine:
    """
    KNN-based pattern similarity engine trained on Oracle-labeled winning setups.

    After fit(), call get_confidence(features_dict) at inference time.
    The engine is also serialized to disk so it can be loaded without retraining.
    """

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors
        self._X_long: Optional[np.ndarray] = None
        self._X_short: Optional[np.ndarray] = None
        self._returns_long: Optional[np.ndarray] = None
        self._returns_short: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df_features: pd.DataFrame, oracle_labels: pd.DataFrame) -> "PatternConfidenceEngine":
        """
        Build the pattern database from oracle-labeled training data.

        Args:
            df_features:  DataFrame with all technical features (aligned index)
            oracle_labels: DataFrame with 'direction_label' and 'expected_return_pct'
        """
        # Align indices
        common_idx = df_features.index.intersection(oracle_labels.index)
        feat = df_features.loc[common_idx]
        labels = oracle_labels.loc[common_idx]

        # Extract available pattern features
        available = [f for f in PATTERN_FEATURES if f in feat.columns]
        if len(available) < 5:
            logger.warning(
                f"PatternConfidenceEngine: only {len(available)} pattern features found. "
                "Confidence scores may be less reliable."
            )
        self._feature_names = available

        X_all = feat[available].values.astype(np.float64)

        # Fit StandardScaler on all non-neutral bars
        active_mask = labels["direction_label"].isin([0, 2])
        X_active = X_all[active_mask]
        self._mean = np.nanmean(X_active, axis=0)
        self._scale = np.nanstd(X_active, axis=0) + 1e-9
        X_scaled = (X_all - self._mean) / self._scale
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Separate LONG and SHORT patterns
        long_mask = labels["direction_label"].values == 0
        short_mask = labels["direction_label"].values == 2

        self._X_long = X_scaled[long_mask]
        self._returns_long = labels["expected_return_pct"].values[long_mask].astype(np.float64)

        self._X_short = X_scaled[short_mask]
        self._returns_short = labels["expected_return_pct"].values[short_mask].astype(np.float64)

        self._fitted = True
        logger.info(
            f"PatternConfidenceEngine fitted: "
            f"{len(self._X_long)} LONG patterns, "
            f"{len(self._X_short)} SHORT patterns, "
            f"{len(available)} features used."
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_confidence(self, features: dict, direction: str) -> float:
        """
        Compute pattern-based confidence for a potential trade.

        Args:
            features:  Dict of {feature_name: value} for the current bar
            direction: "LONG" or "SHORT"

        Returns:
            confidence score 0.0–1.0
              0.0 = current setup looks nothing like past winners
              1.0 = current setup is very similar to highest-return past winners
        """
        if not self._fitted:
            return 0.5  # Neutral fallback if not fitted

        direction = direction.upper()
        X_db = self._X_long if direction == "LONG" else self._X_short
        returns_db = self._returns_long if direction == "LONG" else self._returns_short

        if X_db is None or len(X_db) == 0:
            return 0.5

        # Build query vector
        query = np.array(
            [features.get(f, 0.0) for f in self._feature_names],
            dtype=np.float64,
        )
        query_scaled = (query - self._mean) / self._scale
        query_scaled = np.nan_to_num(query_scaled, nan=0.0)

        # Compute Euclidean distances to all patterns
        diffs = X_db - query_scaled
        distances = np.sqrt((diffs ** 2).sum(axis=1))

        # Take K nearest
        k = min(self.n_neighbors, len(X_db))
        nearest_idx = np.argpartition(distances, k)[:k]
        nearest_distances = distances[nearest_idx]
        nearest_returns = returns_db[nearest_idx]

        # Inverse-distance weighting
        weights = 1.0 / (nearest_distances + 1e-9)
        weighted_return = float(np.average(nearest_returns, weights=weights))

        # Normalize: map expected return to 0-1 confidence
        # 0.5% expected return (max for scalping) → 1.0 confidence
        from config import cfg
        max_return = cfg.testing.MAX_ATR_TARGET_PCT  # e.g. 0.80% for 5m scalping
        confidence = float(np.clip(weighted_return / max(max_return, 0.01), 0.0, 1.0))
        return confidence

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = PATTERN_DB_PATH) -> None:
        """Serialize the fitted engine to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"PatternConfidenceEngine saved to {path}")

    @classmethod
    def load(cls, path: str = PATTERN_DB_PATH) -> "PatternConfidenceEngine":
        """Load a previously saved engine from disk."""
        with open(path, "rb") as f:
            engine = pickle.load(f)
        logger.info(f"PatternConfidenceEngine loaded from {path}")
        return engine

    @classmethod
    def load_or_none(cls, path: str = PATTERN_DB_PATH) -> Optional["PatternConfidenceEngine"]:
        """Load engine if it exists, else return None (graceful fallback)."""
        if not os.path.exists(path):
            logger.warning(f"Pattern confidence DB not found at {path}. Using fixed margin sizing.")
            return None
        try:
            return cls.load(path)
        except Exception as e:
            logger.warning(f"Failed to load pattern confidence DB: {e}. Using fixed margin sizing.")
            return None
