"""TournamentSimulator for the FIFA World Cup Predictor."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from predictor.predictor_api import PredictorAPI


@dataclass
class SimulationResult:
    """Aggregated results from Monte Carlo tournament simulation."""

    win_probabilities: dict[str, float]
    semifinal_probabilities: dict[str, float]
    final_probabilities: dict[str, float]
    ranked_teams: list[str]


class TournamentSimulator:
    """Monte Carlo bracket simulation for a 32-team World Cup tournament."""

    # Standard WC bracket: 1st of group X vs 2nd of group Y
    # Groups A-H, bracket pairings (0-indexed group positions):
    # R16: A1 vs B2, C1 vs D2, E1 vs F2, G1 vs H2,
    #      B1 vs A2, D1 vs C2, F1 vs E2, H1 vs G2
    _BRACKET_PAIRINGS = [
        (0, 1),  # 1st Group A vs 2nd Group B
        (2, 3),  # 1st Group C vs 2nd Group D
        (4, 5),  # 1st Group E vs 2nd Group F
        (6, 7),  # 1st Group G vs 2nd Group H
        (1, 0),  # 1st Group B vs 2nd Group A
        (3, 2),  # 1st Group D vs 2nd Group C
        (5, 4),  # 1st Group F vs 2nd Group E
        (7, 6),  # 1st Group H vs 2nd Group G
    ]

    def __init__(self, predictor: PredictorAPI, n_runs: int = 1000, seed: int = 42):
        if n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {n_runs}")
        self.predictor = predictor
        self.n_runs = n_runs
        self.seed = seed

    def _validate_teams(self, teams: list[str]) -> None:
        """Raise ValueError for any team not known to the predictor."""
        for team in teams:
            try:
                self.predictor._resolve_team(team)
            except ValueError:
                raise ValueError(f"Unknown team name: '{team}'")

    def _simulate_group_stage(
        self, groups: dict[str, list[str]]
    ) -> tuple[list[str], list[str]]:
        """Simulate group stage and return (group_winners, group_runners_up).

        Uses expected win probability approach: for each team, sum win_prob
        across all group matches. Top 2 advance.
        """
        group_winners: list[str] = []
        group_runners_up: list[str] = []

        for group_name in sorted(groups.keys()):
            group_teams = groups[group_name]
            scores: dict[str, float] = {t: 0.0 for t in group_teams}

            for home, away in combinations(group_teams, 2):
                pred = self.predictor.predict(home, away)
                scores[home] += pred.home_win_prob
                scores[away] += pred.away_win_prob

            ranked = sorted(group_teams, key=lambda t: scores[t], reverse=True)
            group_winners.append(ranked[0])
            group_runners_up.append(ranked[1])

        return group_winners, group_runners_up

    def _knockout_winner(self, home: str, away: str, rng: np.random.Generator) -> str:
        """Sample a knockout match winner from the probability distribution."""
        pred = self.predictor.predict(home, away)
        r = rng.random()
        if r < pred.home_win_prob:
            return home
        elif r < pred.home_win_prob + pred.away_win_prob:
            return away
        else:
            # Draw → 50/50 coin flip
            return home if rng.random() < 0.5 else away

    def _simulate_knockout(
        self,
        group_winners: list[str],
        group_runners_up: list[str],
        rng: np.random.Generator,
    ) -> tuple[str, set[str], set[str]]:
        """Run knockout rounds and return (winner, semifinalists, finalists)."""
        # Build R16 matchups using standard WC bracket pairings
        r16_matches: list[tuple[str, str]] = []
        for w_idx, r_idx in self._BRACKET_PAIRINGS:
            r16_matches.append((group_winners[w_idx], group_runners_up[r_idx]))

        # Round of 16 → 8 teams
        qf_teams = [self._knockout_winner(h, a, rng) for h, a in r16_matches]

        # Quarter-finals → 4 teams (semi-finalists)
        sf_teams = [
            self._knockout_winner(qf_teams[i], qf_teams[i + 1], rng)
            for i in range(0, 8, 2)
        ]
        semifinalists = set(sf_teams)

        # Semi-finals → 2 finalists
        finalists_list = [
            self._knockout_winner(sf_teams[0], sf_teams[1], rng),
            self._knockout_winner(sf_teams[2], sf_teams[3], rng),
        ]
        finalists = set(finalists_list)

        # Final → winner
        winner = self._knockout_winner(finalists_list[0], finalists_list[1], rng)

        return winner, semifinalists, finalists

    def simulate(
        self, teams: list[str], groups: dict[str, list[str]]
    ) -> SimulationResult:
        """Run n_runs full bracket simulations and return aggregated probabilities.

        Args:
            teams: List of all participating team names.
            groups: Mapping of group name → list of team names in that group.

        Returns:
            SimulationResult with win/semifinal/final probabilities and ranked teams.

        Raises:
            ValueError: If any team name is not known to the predictor.
        """
        self._validate_teams(teams)

        win_counts: dict[str, int] = {t: 0 for t in teams}
        sf_counts: dict[str, int] = {t: 0 for t in teams}
        final_counts: dict[str, int] = {t: 0 for t in teams}

        rng = np.random.default_rng(self.seed)

        # Group stage is deterministic (expected value approach), compute once
        group_winners, group_runners_up = self._simulate_group_stage(groups)

        for _ in range(self.n_runs):
            winner, semifinalists, finalists = self._simulate_knockout(
                group_winners, group_runners_up, rng
            )
            win_counts[winner] += 1
            for t in semifinalists:
                sf_counts[t] += 1
            for t in finalists:
                final_counts[t] += 1

        # Also count group-stage teams that didn't advance as having 0 counts
        # (already initialised to 0 above)

        win_probs = {t: win_counts[t] / self.n_runs for t in teams}
        sf_probs = {t: sf_counts[t] / self.n_runs for t in teams}
        final_probs = {t: final_counts[t] / self.n_runs for t in teams}

        ranked = sorted(teams, key=lambda t: win_probs[t], reverse=True)

        return SimulationResult(
            win_probabilities=win_probs,
            semifinal_probabilities=sf_probs,
            final_probabilities=final_probs,
            ranked_teams=ranked,
        )
