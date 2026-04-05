"""Tests for ReportGenerator."""

import matplotlib
matplotlib.use('Agg')

import os
import pandas as pd
import pytest

from predictor.reporter import ReportGenerator
from predictor.simulator import SimulationResult


def _make_sim_result(teams):
    """Build a minimal SimulationResult for the given team list."""
    n = len(teams)
    win_probs = {t: 1.0 / n for t in teams}
    sf_probs = {t: 2.0 / n for t in teams}
    final_probs = {t: 1.5 / n for t in teams}
    ranked = list(teams)
    return SimulationResult(
        win_probabilities=win_probs,
        semifinal_probabilities=sf_probs,
        final_probabilities=final_probs,
        ranked_teams=ranked,
    )


def test_charts_saved_as_png(tmp_path):
    """All three chart methods save the expected PNG files."""
    reporter = ReportGenerator(str(tmp_path))

    teams = [f"Team{i}" for i in range(12)]
    sim_result = _make_sim_result(teams)

    reporter.bar_chart(sim_result)
    assert os.path.exists(os.path.join(str(tmp_path), "win_probability_bar_chart.png"))

    y_true = ["Home Win", "Away Win", "Draw", "Home Win"]
    y_pred = ["Home Win", "Home Win", "Draw", "Away Win"]
    reporter.confusion_matrix(y_true, y_pred)
    assert os.path.exists(os.path.join(str(tmp_path), "confusion_matrix.png"))

    importance = pd.Series(
        {f"feature_{i}": float(i) for i in range(25)}
    )
    reporter.feature_importance(importance)
    assert os.path.exists(os.path.join(str(tmp_path), "feature_importance.png"))


def test_summary_csv_columns(tmp_path):
    """summary_csv writes required columns and win_prob values are in [0, 1]."""
    reporter = ReportGenerator(str(tmp_path))

    teams = [f"Team{i}" for i in range(8)]
    sim_result = _make_sim_result(teams)
    reporter.summary_csv(sim_result)

    csv_path = os.path.join(str(tmp_path), "simulation_summary.csv")
    assert os.path.exists(csv_path)

    df = pd.read_csv(csv_path)
    assert set(["team", "win_prob", "semifinal_prob", "final_prob"]).issubset(df.columns)
    assert df["win_prob"].between(0, 1).all()
