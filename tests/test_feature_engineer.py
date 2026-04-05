"""Unit tests for FeatureEngineer."""

import pandas as pd
import pytest

from predictor.feature_engineer import FeatureEngineer
from predictor.config import FEATURE_COLS, TARGET_COL


def test_build_features_returns_dataframe(matches_df, players_df):
    """build_features() returns a DataFrame with the expected columns."""
    fe = FeatureEngineer(matches_df, players_df)  # results_df optional, falls back to WC data
    result = fe.build_features()

    assert isinstance(result, pd.DataFrame), "build_features() should return a DataFrame"

    expected_cols = set(FEATURE_COLS) | {TARGET_COL}
    actual_cols = set(result.columns)
    assert expected_cols.issubset(actual_cols), (
        f"Missing columns: {expected_cols - actual_cols}"
    )
