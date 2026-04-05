"""Unit tests for DataLoader."""

import pytest
from predictor.data_loader import DataLoader
from predictor.config import MATCHES_REQUIRED_COLS, PLAYERS_REQUIRED_COLS, CUPS_REQUIRED_COLS


def test_dataloader_loads_all_files(tmp_data_dir):
    """DataLoader.load() returns four non-empty DataFrames with expected columns."""
    loader = DataLoader(tmp_data_dir)
    matches_df, players_df, cups_df, results_df = loader.load()

    assert len(matches_df) > 0
    assert len(players_df) > 0
    assert len(cups_df) > 0
    assert len(results_df) > 0

    for col in ["Year", "Home Team Name", "Away Team Name", "Home Team Goals", "Away Team Goals", "MatchID", "Stage", "Datetime"]:
        assert col in matches_df.columns, f"Missing column in matches_df: {col}"

    for col in ["MatchID", "Team Initials", "Player Name", "Event"]:
        assert col in players_df.columns, f"Missing column in players_df: {col}"

    for col in ["Year", "Winner"]:
        assert col in cups_df.columns, f"Missing column in cups_df: {col}"

    for col in ["date", "home_team", "away_team", "home_score", "away_score"]:
        assert col in results_df.columns, f"Missing column in results_df: {col}"
