"""Unit tests for TournamentSimulator."""

from unittest.mock import MagicMock

import pytest

from predictor.predictor_api import MatchPrediction
from predictor.simulator import SimulationResult, TournamentSimulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_equal_predictor(known_teams: list[str]) -> MagicMock:
    """Return a mock PredictorAPI that always returns equal probabilities (1/3 each)."""
    mock = MagicMock()

    def _resolve_team(name: str) -> str:
        lower_map = {t.lower(): t for t in known_teams}
        canonical = lower_map.get(name.lower())
        if canonical is None:
            raise ValueError(f"Unrecognised team name: '{name}'")
        return canonical

    mock._resolve_team.side_effect = _resolve_team
    mock.predict.return_value = MatchPrediction(
        home_win_prob=1 / 3,
        away_win_prob=1 / 3,
        draw_prob=1 / 3,
        predicted_label="Draw",
    )
    return mock


def _make_32_teams_and_groups() -> tuple[list[str], dict[str, list[str]]]:
    """Generate 32 team names in 8 groups of 4."""
    group_names = list("ABCDEFGH")
    teams = [f"Team_{g}{i}" for g in group_names for i in range(1, 5)]
    groups = {
        g: [f"Team_{g}{i}" for i in range(1, 5)] for g in group_names
    }
    return teams, groups


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_simulator_accepts_32_teams():
    """TournamentSimulator.simulate() should run without error for 32 teams in 8 groups of 4."""
    teams, groups = _make_32_teams_and_groups()
    predictor = _make_equal_predictor(teams)
    simulator = TournamentSimulator(predictor, n_runs=10, seed=42)
    result = simulator.simulate(teams, groups)
    assert isinstance(result, SimulationResult)


def test_win_probs_sum_to_one():
    """win_probabilities should sum to 1.0 ± 1e-4."""
    teams, groups = _make_32_teams_and_groups()
    predictor = _make_equal_predictor(teams)
    simulator = TournamentSimulator(predictor, n_runs=100, seed=42)
    result = simulator.simulate(teams, groups)
    total = sum(result.win_probabilities.values())
    assert abs(total - 1.0) < 1e-4


def test_invalid_n_runs_raises():
    """TournamentSimulator should raise ValueError when n_runs < 1."""
    predictor = MagicMock()
    with pytest.raises(ValueError):
        TournamentSimulator(predictor, n_runs=0)


def test_unknown_team_raises_value_error():
    """simulate() should raise ValueError for unknown team names before simulation starts."""
    teams, groups = _make_32_teams_and_groups()
    predictor = _make_equal_predictor(teams)
    simulator = TournamentSimulator(predictor, n_runs=10, seed=42)

    bad_teams = teams[:-1] + ["UnknownTeam"]
    bad_groups = dict(groups)
    bad_groups["H"] = bad_groups["H"][:-1] + ["UnknownTeam"]

    with pytest.raises(ValueError, match="UnknownTeam"):
        simulator.simulate(bad_teams, bad_groups)


def test_ranked_teams_contains_all_teams():
    """ranked_teams should contain all 32 input teams exactly once."""
    teams, groups = _make_32_teams_and_groups()
    predictor = _make_equal_predictor(teams)
    simulator = TournamentSimulator(predictor, n_runs=50, seed=42)
    result = simulator.simulate(teams, groups)
    assert sorted(result.ranked_teams) == sorted(teams)
    assert len(result.ranked_teams) == len(teams)
