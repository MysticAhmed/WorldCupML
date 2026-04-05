"""Model training for the FIFA World Cup Predictor."""

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
    """Trains and evaluates the gradient boosting match outcome classifier."""

    def __init__(self, features_df: pd.DataFrame, cutoff_year: int, model_path: str):
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
        """Temporal split, hyperparameter search, fit, evaluate."""
        train_df = self.features_df[self.features_df["year"] < self.cutoff_year]
        test_df = self.features_df[self.features_df["year"] >= self.cutoff_year]

        if train_df.empty:
            raise ValueError("Training set is empty after temporal split.")
        if test_df.empty:
            raise ValueError("Test set is empty after temporal split.")

        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL].map(OUTCOME_LABEL_MAP)

        X_test = test_df[FEATURE_COLS]
        y_test = test_df[TARGET_COL].map(OUTCOME_LABEL_MAP)

        # Class weights to handle draw imbalance
        sample_weights = compute_sample_weight("balanced", y_train)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("xgb", XGBClassifier(
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=RANDOM_SEED,
                tree_method="hist",
            )),
        ])

        n_splits = min(5, max(3, len(y_train) // 1000))
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=_PARAM_GRID,
            scoring="f1_weighted",
            n_iter=40,
            cv=n_splits,
            random_state=RANDOM_SEED,
            refit=True,
            n_jobs=-1,
        )
        search.fit(X_train, y_train, xgb__sample_weight=sample_weights)

        best_pipeline: Pipeline = search.best_estimator_

        y_pred = best_pipeline.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

        xgb_step: XGBClassifier = best_pipeline.named_steps["xgb"]
        importance_values = xgb_step.feature_importances_
        # Get the feature names that survived imputation (imputer may drop all-NaN cols)
        imputer = best_pipeline.named_steps["imputer"]
        try:
            surviving_features = list(imputer.get_feature_names_out())
        except Exception:
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
