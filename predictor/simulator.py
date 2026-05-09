"""TournamentSimulator for the 2026 FIFA World Cup (48-team format).

This module simulates the entire 2026 World Cup tournament using Monte Carlo methods:

1. **Group Stage**: 12 groups of 4 teams (Groups A-L)
   - Each team plays 3 matches (round-robin within group)
   - Top 2 from each group advance (24 teams)
   - Best 8 third-place teams also advance (8 teams)
   - Total: 32 teams advance to knockout stage

2. **Knockout Stage**: Single-elimination bracket
   - Round of 32 (16 matches)
   - Round of 16 (8 matches)
   - Quarter-finals (4 matches)
   - Semi-finals (2 matches)
   - Final (1 match)

3. **Stochastic Simulation**: Each match outcome is sampled from predicted probabilities
   - NOT deterministic (highest probability doesn't always win)
   - Captures realistic upset potential
   - Run 1000+ times to get stable probability estimates

4. **Official Bracket**: Uses FIFA's official 2026 bracket structure
   - Third-place team assignments follow FIFA rules
   - Bracket paths prevent early rematches of group opponents

The simulator outputs win probabilities for each team by counting how many times
they win the tournament across all simulation runs.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """Monte Carlo bracket simulation for the 2026 48-team World Cup.

    Format:
    - 12 groups of 4 (Groups A-L)
    - Top 2 from each group + best 8 third-place teams = 32 teams
    - Round of 32 → Round of 16 → QF → SF → Final

    Round of 32 bracket (official FIFA pairings):
        M73:  2nd A  vs 2nd B
        M74:  1st E  vs best 3rd (A/B/C/D/F)
        M75:  1st F  vs 2nd C
        M76:  1st C  vs 2nd F
        M77:  1st I  vs best 3rd (C/D/F/G/H)
        M78:  2nd E  vs 2nd I
        M79:  1st A  vs best 3rd (C/E/F/H/I)
        M80:  1st L  vs best 3rd (E/H/I/J/K)
        M81:  1st D  vs best 3rd (B/E/F/I/J)
        M82:  1st G  vs best 3rd (A/E/H/I/J)
        M83:  2nd K  vs 2nd L
        M84:  1st H  vs 2nd J
        M85:  1st B  vs best 3rd (E/F/G/I/J)
        M86:  1st J  vs 2nd H
        M87:  1st K  vs best 3rd (D/E/I/J/L)
        M88:  2nd D  vs 2nd G

    Round of 16 pairings (winners of R32 matches):
        M89: W74 vs W77   M90: W73 vs W75
        M91: W76 vs W78   M92: W79 vs W80  (note: W79=1stA side, W80=1stL side)
        M93: W84 vs W81   M94: W82 vs W88  (note: corrected per bracket)
        M95: W88 vs W81 ... using official bracket order below

    Official R16 pairings from bracket:
        M89: W74 vs W77
        M90: W73 vs W75
        M91: W76 vs W78
        M92: W79 vs W80  -- actually W79 vs W80 per bracket
        M93: W84 vs W81  -- actually per bracket tree
        M94: W82 vs W88  -- actually per bracket tree
        M95: W83 vs W85  -- actually per bracket tree
        M96: W86 vs W87  -- actually per bracket tree
    """

    # Official R32 bracket as (slot_description) — we'll build dynamically
    # Third-place group pools per match slot
    _THIRD_PLACE_POOLS = {
        "M74": ["A", "B", "C", "D", "F"],
        "M77": ["C", "D", "F", "G", "H"],
        "M79": ["C", "E", "F", "H", "I"],
        "M80": ["E", "H", "I", "J", "K"],
        "M81": ["B", "E", "F", "I", "J"],
        "M82": ["A", "E", "H", "I", "J"],
        "M85": ["E", "F", "G", "I", "J"],
        "M87": ["D", "E", "I", "J", "L"],
    }

    def __init__(self, predictor: PredictorAPI, n_runs: int = 1000, seed: int = 42):
        if n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {n_runs}")
        self.predictor = predictor
        self.n_runs = n_runs
        self.seed = seed

    def _validate_teams(self, teams: list[str]) -> None:
        for team in teams:
            try:
                self.predictor._resolve_team(team)
            except ValueError:
                raise ValueError(f"Unknown team name: '{team}'")

    def _simulate_group_stage(
        self, groups: dict[str, list[str]]
    ) -> tuple[dict[str, str], dict[str, str], dict[str, tuple[str, float]]]:
        """Simulate group stage with stochastic match outcomes.

        Each match is sampled from predicted probabilities rather than using
        expected points, giving realistic upset potential.
        
        For example, if Brazil vs Switzerland has probabilities:
        - Brazil win: 60%
        - Draw: 25%
        - Switzerland win: 15%
        
        We sample from this distribution (not just pick Brazil).
        This means Switzerland can win 15% of simulations, creating realistic upsets.
        
        Returns:
            (winners, runners_up, third_place)
            - winners: group_name -> 1st place team
            - runners_up: group_name -> 2nd place team
            - third_place: group_name -> (3rd place team, points earned)
        """
        rng = np.random.default_rng()  # Fresh RNG per call for stochasticity
        winners: dict[str, str] = {}
        runners_up: dict[str, str] = {}
        third_place: dict[str, tuple[str, float]] = {}

        for group_name in sorted(groups.keys()):
            group_teams = groups[group_name]
            points: dict[str, int] = {t: 0 for t in group_teams}  # 3 for win, 1 for draw, 0 for loss
            gd: dict[str, float] = {t: 0.0 for t in group_teams}  # Goal difference (tiebreaker)

            # Play all pairwise matches (round-robin)
            for home, away in combinations(group_teams, 2):
                pred = self.predictor.predict(home, away)
                
                # Sample outcome from probability distribution
                r = rng.random()  # Random number in [0, 1)
                if r < pred.home_win_prob:
                    # Home team wins
                    points[home] += 3
                    gd[home] += 1; gd[away] -= 1
                elif r < pred.home_win_prob + pred.draw_prob:
                    # Draw
                    points[home] += 1; points[away] += 1
                else:
                    # Away team wins
                    points[away] += 3
                    gd[away] += 1; gd[home] -= 1

            # Rank teams by points, then goal difference
            ranked = sorted(group_teams, key=lambda t: (points[t], gd[t]), reverse=True)
            winners[group_name] = ranked[0]
            runners_up[group_name] = ranked[1]
            third_place[group_name] = (ranked[2], float(points[ranked[2]]))

        return winners, runners_up, third_place

    def _pick_best_third(
        self,
        third_place: dict[str, tuple[str, float]],
        pool: list[str],
        already_used: set[str],
    ) -> str:
        """Pick the best available 3rd-place team from the given group pool."""
        candidates = [
            (g, third_place[g])
            for g in pool
            if g in third_place and third_place[g][0] not in already_used
        ]
        if not candidates:
            # Fallback: pick any unused 3rd place team
            candidates = [
                (g, third_place[g])
                for g in third_place
                if third_place[g][0] not in already_used
            ]
        best_group = max(candidates, key=lambda x: x[1][1])
        team = best_group[1][0]
        already_used.add(team)
        return team

    def _knockout_winner(self, home: str, away: str, rng: np.random.Generator) -> str:
        """Sample a knockout match winner.
        
        In knockout matches, draws go to extra time and penalties.
        We simplify by treating draws as 50/50 coin flips.
        This is reasonable because extra time and penalties are highly unpredictable.
        """
        pred = self.predictor.predict(home, away)
        r = rng.random()
        if r < pred.home_win_prob:
            return home
        elif r < pred.home_win_prob + pred.away_win_prob:
            return away
        else:
            # Draw: flip a coin to determine winner
            return home if rng.random() < 0.5 else away

    def _simulate_knockout(
        self,
        winners: dict[str, str],
        runners_up: dict[str, str],
        third_place: dict[str, tuple[str, float]],
        rng: np.random.Generator,
    ) -> tuple[str, set[str], set[str]]:
        """Run Round of 32 through Final. Returns (winner, semifinalists, finalists)."""
        used_thirds: set[str] = set()

        def w(g): return winners[g]
        def r(g): return runners_up[g]
        def t(pool): return self._pick_best_third(third_place, pool, used_thirds)

        # ── Round of 32 ───────────────────────────────────────────────────────
        m73  = self._knockout_winner(r("A"), r("B"), rng)
        m74  = self._knockout_winner(w("E"), t(["A","B","C","D","F"]), rng)
        m75  = self._knockout_winner(w("F"), r("C"), rng)
        m76  = self._knockout_winner(w("C"), r("F"), rng)
        m77  = self._knockout_winner(w("I"), t(["C","D","F","G","H"]), rng)
        m78  = self._knockout_winner(r("E"), r("I"), rng)
        m79  = self._knockout_winner(w("A"), t(["C","E","F","H","I"]), rng)
        m80  = self._knockout_winner(w("L"), t(["E","H","I","J","K"]), rng)
        m81  = self._knockout_winner(w("D"), t(["B","E","F","I","J"]), rng)
        m82  = self._knockout_winner(w("G"), t(["A","E","H","I","J"]), rng)
        m83  = self._knockout_winner(r("K"), r("L"), rng)
        m84  = self._knockout_winner(w("H"), r("J"), rng)
        m85  = self._knockout_winner(w("B"), t(["E","F","G","I","J"]), rng)
        m86  = self._knockout_winner(w("J"), r("H"), rng)
        m87  = self._knockout_winner(w("K"), t(["D","E","I","J","L"]), rng)
        m88  = self._knockout_winner(r("D"), r("G"), rng)

        # ── Round of 16 ───────────────────────────────────────────────────────
        m89 = self._knockout_winner(m74, m77, rng)
        m90 = self._knockout_winner(m73, m75, rng)
        m91 = self._knockout_winner(m76, m78, rng)
        m92 = self._knockout_winner(m79, m80, rng)
        m93 = self._knockout_winner(m84, m81, rng)
        m94 = self._knockout_winner(m82, m88, rng)
        m95 = self._knockout_winner(m83, m85, rng)
        m96 = self._knockout_winner(m86, m87, rng)

        # ── Quarter-finals ────────────────────────────────────────────────────
        m97 = self._knockout_winner(m89, m90, rng)
        m98 = self._knockout_winner(m91, m92, rng)
        m99 = self._knockout_winner(m93, m94, rng)
        m100 = self._knockout_winner(m95, m96, rng)

        # ── Semi-finals ───────────────────────────────────────────────────────
        m101_a = self._knockout_winner(m97, m98, rng)
        m101_b = self._knockout_winner(m99, m100, rng)
        semifinalists = {m97, m98, m99, m100}

        # ── Final ─────────────────────────────────────────────────────────────
        finalists = {m101_a, m101_b}
        winner = self._knockout_winner(m101_a, m101_b, rng)

        return winner, semifinalists, finalists

    def simulate(
        self, teams: list[str], groups: dict[str, list[str]]
    ) -> SimulationResult:
        """Run n_runs full tournament simulations.

        Args:
            teams:  All participating team names.
            groups: group_name -> list of team names (12 groups for 2026 WC).

        Returns:
            SimulationResult with win/semifinal/final probabilities.
        """
        self._validate_teams(teams)

        win_counts:   dict[str, int] = {t: 0 for t in teams}
        sf_counts:    dict[str, int] = {t: 0 for t in teams}
        final_counts: dict[str, int] = {t: 0 for t in teams}

        rng = np.random.default_rng(self.seed)

        # Group stage is now stochastic — run inside the loop
        for _ in range(self.n_runs):
            winners, runners_up, third_place = self._simulate_group_stage(groups)
            winner, semifinalists, finalists = self._simulate_knockout(
                winners, runners_up, third_place, rng
            )
            win_counts[winner] += 1
            for t in semifinalists:
                sf_counts[t] += 1
            for t in finalists:
                final_counts[t] += 1

        win_probs   = {t: win_counts[t]   / self.n_runs for t in teams}
        sf_probs    = {t: sf_counts[t]    / self.n_runs for t in teams}
        final_probs = {t: final_counts[t] / self.n_runs for t in teams}

        ranked = sorted(teams, key=lambda t: win_probs[t], reverse=True)

        return SimulationResult(
            win_probabilities=win_probs,
            semifinal_probabilities=sf_probs,
            final_probabilities=final_probs,
            ranked_teams=ranked,
        )
