"""Unit tests for PredictorAPI."""

import pandas as pd
import pytest

from predictor.feature_engineer import FeatureEngineer
from predictor.model_trainer import ModelTrainer
from predictor.predictor_api import MatchPrediction, PredictorAPI


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
def feature_engineer(matches_df_with_draw, players_df):
    """Build a FeatureEngineer from the sample conftest fixtures."""
    return FeatureEngineer(matches_df_with_draw, players_df)


@pytest.fixture
def predictor_api(feature_engineer):
    """Train a model and return a PredictorAPI instance."""
    features_df = feature_engineer.build_features()
    trainer = ModelTrainer(features_df, cutoff_year=1966, model_path="models/test")
    result = trainer.train()
    return PredictorAPI(result.model, feature_engineer)


def test_predict_returns_match_prediction(predictor_api):
    """predict() should return a MatchPrediction with probabilities summing to 1.0 ± 1e-6."""
    prediction = predictor_api.predict("France", "Brazil")

    assert isinstance(prediction, MatchPrediction)
    total = prediction.home_win_prob + prediction.away_win_prob + prediction.draw_prob
    assert abs(total - 1.0) < 1e-6


def test_unknown_team_raises_value_error(predictor_api):
    """predict() should raise ValueError for an unknown team name."""
    with pytest.raises(ValueError, match="UnknownTeamXYZ"):
        predictor_api.predict("UnknownTeamXYZ", "Brazil")

    with pytest.raises(ValueError, match="AlsoUnknown"):
        predictor_api.predict("France", "AlsoUnknown")
