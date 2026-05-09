"""Model training for the FIFA World Cup Predictor.

This module handles:
1. Temporal train/test split (train on past, test on future)
2. Hyperparameter tuning via RandomizedSearchCV
3. Class imbalance handling (draws are rare, ~20% of matches)
4. Model evaluation with multiple metrics
5. Feature importance extraction

We use XGBoost (gradient boosting) because:
- Handles non-linear relationships well (e.g., ELO difference → win probability)
- Robust to missing values and outliers
- Provides feature importance scores
- Fast training even with 28 features and 1000+ samples
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from predictor.config import FEATURE_COLS, OUTCOME_LABEL_MAP, RANDOM_SEED, TARGET_COL

MODEL_FILENAME = "match_outcome_model.joblib"

# Hyperparameter search space for XGBoost
# These ranges were chosen based on common best practices:
# - n_estimators: more trees = better fit but slower training
# - max_depth: deeper trees = more complex patterns but risk overfitting
# - learning_rate: lower = more conservative updates, needs more trees
# - subsample: fraction of samples per tree (< 1.0 adds randomness, reduces overfitting)
# - colsample_bytree: fraction of features per tree (prevents feature dominance)
# - min_child_weight: minimum samples per leaf (higher = more conservative)
# - gamma: minimum loss reduction to split (higher = more conservative)
_PARAM_GRID = {
    "xgb__n_estimators": [200, 400, 600],
    "xgb__max_depth": [3, 4, 5, 6],
    "xgb__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "xgb__subsample": [0.7, 0.8, 0.9, 1.0],
    "xgb__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "xgb__min_child_weight": [1, 3, 5],
    "xgb__gamma": [0, 0.1, 0.3],
}


@dataclass
class TrainResult:
    """Result of a training run."""

    model: Pipeline
    metrics: dict
    feature_importance: pd.Series


class ModelTrainer:
    """Trains and evaluates the gradient boosting match outcome classifier.
    
    Training pipeline:
    1. Temporal split: train on matches before cutoff_year, test on matches after
    2. Imputation: fill missing values with column means (handles cold-start teams)
    3. Hyperparameter search: try 40 random combinations, pick best by F1 score
    4. Class weighting: balance training to handle draw underrepresentation
    5. Evaluation: compute accuracy, precision, recall, F1 on held-out test set
    """

    def __init__(self, features_df: pd.DataFrame, cutoff_year: int, model_path: str):
        # Validate cutoff year is within data range
        min_year = int(features_df["year"].min())
        max_year = int(features_df["year"].max())
        if cutoff_year < min_year or cutoff_year > max_year:
            raise ValueError(
                f"cutoff_year {cutoff_year} is outside the data year range "
                f"[{min_year}, {max_year}]"
            )
        self.features_df = features_df
        self.cutoff_year = cutoff_year
        self.model_path = model_path
        self._pipeline: Pipeline | None = None

    def train(self) -> TrainResult:
        """Temporal split, hyperparameter search, fit, evaluate.
        
        Returns TrainResult with trained model, evaluation metrics, and feature importance.
        """
        # CRITICAL: Temporal split prevents data leakage
        # Train on past (< cutoff_year), test on future (>= cutoff_year)
        # This simulates real-world prediction: we can't use future data to predict past
        train_df = self.features_df[self.features_df["year"] < self.cutoff_year]
        test_df = self.features_df[self.features_df["year"] >= self.cutoff_year]

        if train_df.empty:
            raise ValueError("Training set is empty after temporal split.")
        if test_df.empty:
            raise ValueError("Test set is empty after temporal split.")

        # Separate features (X) from target (y)
        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL].map(OUTCOME_LABEL_MAP)  # "Home Win" → 0, "Draw" → 1, "Away Win" → 2

        X_test = test_df[FEATURE_COLS]
        y_test = test_df[TARGET_COL].map(OUTCOME_LABEL_MAP)

        # Compute sample weights to handle class imbalance
        # Draws are ~20% of matches, wins are ~40% each
        # Balanced weighting gives draws 2x weight to compensate
        sample_weights = compute_sample_weight("balanced", y_train)

        # Build sklearn pipeline: imputation → XGBoost
        # Pipeline ensures imputation is fit on training data only (no leakage)
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),  # Fill NaN with column mean
            ("xgb", XGBClassifier(
                use_label_encoder=False,  # Deprecated parameter, disable warning
                eval_metric="mlogloss",   # Multi-class log loss
                random_state=RANDOM_SEED,
                tree_method="hist",       # Faster histogram-based algorithm
            )),
        ])

        # Adaptive cross-validation: use 3-5 folds depending on training set size
        # Smaller datasets need fewer folds to ensure each fold has enough samples
        n_splits = min(5, max(3, len(y_train) // 1000))
        
        # RandomizedSearchCV: try 40 random hyperparameter combinations
        # Faster than GridSearchCV (which tries all combinations)
        # Scoring by F1 (weighted) balances precision and recall across all 3 classes
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=_PARAM_GRID,
            scoring="f1_weighted",
            n_iter=40,  # Try 40 random combinations
            cv=n_splits,
            random_state=RANDOM_SEED,
            refit=True,  # Refit best model on full training set
            n_jobs=-1,   # Use all CPU cores
        )
        search.fit(X_train, y_train, xgb__sample_weight=sample_weights)

        # Extract best model from search
        best_pipeline: Pipeline = search.best_estimator_

        # Evaluate on held-out test set
        y_pred = best_pipeline.predict(X_test)

        # Compute multiple metrics for comprehensive evaluation
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),  # Overall correctness
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),  # Avoid false positives
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),  # Avoid false negatives
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),  # Harmonic mean of precision/recall
        }

        # Extract feature importance from trained XGBoost model
        # Higher values = more important for predictions
        xgb_step: XGBClassifier = best_pipeline.named_steps["xgb"]
        importance_values = xgb_step.feature_importances_
        
        # Get feature names after imputation (imputer may drop all-NaN columns)
        imputer = best_pipeline.named_steps["imputer"]
        try:
            surviving_features = list(imputer.get_feature_names_out())
        except Exception:
            # Fallback if get_feature_names_out() not available
            surviving_features = FEATURE_COLS[: len(importance_values)]
        feature_importance = pd.Series(importance_values, index=surviving_features)

        self._pipeline = best_pipeline

        return TrainResult(
            model=best_pipeline,
            metrics=metrics,
            feature_importance=feature_importance,
        )

    def save(self):
        """Save the fitted pipeline to <model_path>/match_outcome_model.joblib."""
        if self._pipeline is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        os.makedirs(self.model_path, exist_ok=True)
        joblib.dump(self._pipeline, os.path.join(self.model_path, MODEL_FILENAME))

    @staticmethod
    def load(model_path: str) -> Pipeline:
        """Load a fitted pipeline from <model_path>/match_outcome_model.joblib."""
        filepath = os.path.join(model_path, MODEL_FILENAME)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)
