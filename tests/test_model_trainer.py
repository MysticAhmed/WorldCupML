"""Unit tests for ModelTrainer."""

import pandas as pd
import pytest

from predictor.feature_engineer import FeatureEngineer
from predictor.model_trainer import ModelTrainer


@pytest.fixture
def matches_df_with_draw(matches_df):
    """Extend the sample matches_df with a draw so all 3 outcome classes exist."""
    draw_row = pd.DataFrame([{
        "MatchID": 13,
        "Year": 1950,
        "Datetime": "25 Jun 1950 - 15:00",
        "Stage": "Group 1",
        "Home Team Name": "France",
        "Away Team Name": "England",
        "Home Team Goals": 2,
        "Away Team Goals": 2,
    }])
    return pd.concat([matches_df, draw_row], ignore_index=True)


@pytest.fixture
def features_df(matches_df_with_draw, players_df):
    """Build a features DataFrame from the sample conftest fixtures."""
    fe = FeatureEngineer(matches_df_with_draw, players_df)
    return fe.build_features()


@pytest.fixture
def trained_result(features_df):
    """Train a ModelTrainer on the sample features and return the TrainResult."""
    # Data spans 1930-1974; use 1966 so both splits are non-empty and train has all 3 classes
    trainer = ModelTrainer(features_df, cutoff_year=1966, model_path="models/test")
    return trainer.train()


def test_model_has_three_classes(trained_result):
    """The trained model should predict exactly 3 outcome classes."""
    model = trained_result.model
    assert len(model.classes_) == 3


def test_metrics_keys(trained_result):
    """The metrics dict must contain accuracy, precision, recall, and f1."""
    metrics = trained_result.metrics
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
