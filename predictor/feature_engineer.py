"""Feature engineering using the full international results dataset + ELO ratings."""

from __future__ import annotations

import functools
import numpy as np
import pandas as pd

from predictor.config import (
    CURRENT_ELO_RATINGS,
    DEFAULT_TOURNAMENT_WEIGHT,
    ELO_HOME_ADVANTAGE,
    ELO_INITIAL_RATING,
    ELO_K_BASE,
    FEATURE_COLS,
    STAGE_ORDINAL_MAP,
    TARGET_COL,
    TOURNAMENT_WEIGHT,
)


class FeatureEngineer:
    """Derives Team_Feature_Vector for every match row.

    Uses results_df (45k+ international matches) for rolling stats and ELO,
    and matches_df (WC only) for WC-specific features and training targets.
    """

    def __init__(
        self,
        matches_df: pd.DataFrame,
        players_df: pd.DataFrame,
        results_df: pd.DataFrame | None = None,
    ):
        # Clean matches_df: ensure Year is valid
        matches_df = matches_df.dropna(subset=["Year"]).copy()
        matches_df["Year"] = matches_df["Year"].astype(int)
        
        self.matches_df = matches_df
        self.players_df = players_df.copy()

        # Use international results if provided, otherwise fall back to WC matches only
        if results_df is not None and not results_df.empty:
            self.results_df = results_df.copy()
        else:
            # Build a results_df-compatible frame from WC matches
            self.results_df = self._wc_to_results(self.matches_df)

        # Pre-compute ELO timeline from the full results history
        self._elo_timeline: dict[str, dict[int, float]] = {}  # team -> {match_index -> elo}
        self._elo_by_date: pd.DataFrame = pd.DataFrame()
        self._compute_elo_timeline()

        # Build team-name -> initials map for player aggregates
        self._global_name_initials: dict[str, str] = {}
        self._build_global_name_initials()

        # Caches for expensive per-team stat lookups
        self._rolling_stats_cache: dict[tuple, dict] = {}
        self._h2h_cache: dict[tuple, dict] = {}
        self._player_agg_cache: dict[tuple, float] = {}
        self._build_global_name_initials()

    # ── ELO computation ───────────────────────────────────────────────────────

    def _compute_elo_timeline(self):
        """Compute ELO rating for every team after every match in results_df."""
        df = self.results_df.sort_values("date").reset_index(drop=True)

        ratings: dict[str, float] = {}

        records = []
        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            home_score = row["home_score"]
            away_score = row["away_score"]
            neutral = bool(row.get("neutral", False))
            tournament = str(row.get("tournament", "Friendly"))
            date = row["date"]
            year_raw = row["year"]
            if pd.isna(year_raw):
                continue
            year = int(year_raw)

            r_home = ratings.get(home, ELO_INITIAL_RATING)
            r_away = ratings.get(away, ELO_INITIAL_RATING)

            # Store pre-match ratings
            records.append({
                "date": date,
                "year": year,
                "home_team": home,
                "away_team": away,
                "home_elo_pre": r_home,
                "away_elo_pre": r_away,
            })

            # Update ratings
            r_home_new, r_away_new = self._elo_update(
                r_home, r_away, home_score, away_score, neutral, tournament
            )
            ratings[home] = r_home_new
            ratings[away] = r_away_new

        self._elo_by_date = pd.DataFrame(records)
        # Store final ratings (used for prediction on unseen teams)
        self._final_elo = dict(ratings)

    @staticmethod
    def _elo_update(
        r_home: float,
        r_away: float,
        home_score: float,
        away_score: float,
        neutral: bool,
        tournament: str,
    ) -> tuple[float, float]:
        """Standard ELO update with home advantage and tournament weighting."""
        advantage = 0 if neutral else ELO_HOME_ADVANTAGE
        expected_home = 1 / (1 + 10 ** ((r_away - (r_home + advantage)) / 400))
        expected_away = 1 - expected_home

        if home_score > away_score:
            actual_home, actual_away = 1.0, 0.0
        elif home_score < away_score:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        weight = TOURNAMENT_WEIGHT.get(tournament, DEFAULT_TOURNAMENT_WEIGHT)
        k = ELO_K_BASE * weight

        # Goal difference multiplier (capped at 3)
        gd = abs(home_score - away_score)
        gd_mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else min(1.75 + (gd - 3) * 0.05, 2.0))

        r_home_new = r_home + k * gd_mult * (actual_home - expected_home)
        r_away_new = r_away + k * gd_mult * (actual_away - expected_away)
        return r_home_new, r_away_new

    def get_elo_before(self, team: str, date: pd.Timestamp) -> float:
        """Return the ELO rating for a team just before a given date."""
        if self._elo_by_date.empty:
            return CURRENT_ELO_RATINGS.get(team, ELO_INITIAL_RATING)

        # Rows where this team played as home or away, strictly before date
        mask = (
            (
                (self._elo_by_date["home_team"] == team) |
                (self._elo_by_date["away_team"] == team)
            ) &
            (self._elo_by_date["date"] < date)
        )
        prior = self._elo_by_date[mask]
        if prior.empty:
            return CURRENT_ELO_RATINGS.get(team, ELO_INITIAL_RATING)

        last = prior.iloc[-1]
        if last["home_team"] == team:
            return float(last["home_elo_pre"])
        return float(last["away_elo_pre"])

    # ── Rolling stats (from full international results) ───────────────────────

    def _rolling_team_stats(self, team: str, year: int) -> dict:
        """Rolling win/draw/loss/goals stats using all international matches before year."""
        key = (team, year)
        if key in self._rolling_stats_cache:
            return self._rolling_stats_cache[key]

        df = self.results_df
        home_mask = (df["home_team"] == team) & (df["year"] < year)
        away_mask = (df["away_team"] == team) & (df["year"] < year)

        home_m = df[home_mask]
        away_m = df[away_mask]

        records = []
        for _, r in home_m.iterrows():
            gs, gc = r["home_score"], r["away_score"]
            records.append({
                "gs": gs, "gc": gc,
                "result": "win" if gs > gc else ("loss" if gs < gc else "draw"),
            })
        for _, r in away_m.iterrows():
            gs, gc = r["away_score"], r["home_score"]
            records.append({
                "gs": gs, "gc": gc,
                "result": "win" if gs > gc else ("loss" if gs < gc else "draw"),
            })

        if not records:
            result = {k: np.nan for k in [
                "win_rate", "draw_rate", "loss_rate",
                "avg_goals_scored", "avg_goals_conceded", "avg_goal_diff",
            ]}
            self._rolling_stats_cache[key] = result
            return result

        n = len(records)
        wins = sum(1 for r in records if r["result"] == "win")
        draws = sum(1 for r in records if r["result"] == "draw")
        losses = n - wins - draws
        avg_gs = sum(r["gs"] for r in records) / n
        avg_gc = sum(r["gc"] for r in records) / n
        result = {
            "win_rate": wins / n,
            "draw_rate": draws / n,
            "loss_rate": losses / n,
            "avg_goals_scored": avg_gs,
            "avg_goals_conceded": avg_gc,
            "avg_goal_diff": avg_gs - avg_gc,
        }
        self._rolling_stats_cache[key] = result
        return result

    def _head_to_head_stats(self, team_a: str, team_b: str, year: int) -> dict:
        """H2H stats from all international matches before year."""
        key = (team_a, team_b, year)
        if key in self._h2h_cache:
            return self._h2h_cache[key]

        df = self.results_df
        mask_ab = (df["home_team"] == team_a) & (df["away_team"] == team_b) & (df["year"] < year)
        mask_ba = (df["home_team"] == team_b) & (df["away_team"] == team_a) & (df["year"] < year)

        records = []
        for _, r in df[mask_ab].iterrows():
            gs_a, gs_b = r["home_score"], r["away_score"]
            records.append({
                "gd": gs_a - gs_b,
                "result": "a_win" if gs_a > gs_b else ("b_win" if gs_a < gs_b else "draw"),
            })
        for _, r in df[mask_ba].iterrows():
            gs_a, gs_b = r["away_score"], r["home_score"]
            records.append({
                "gd": gs_a - gs_b,
                "result": "a_win" if gs_a > gs_b else ("b_win" if gs_a < gs_b else "draw"),
            })

        if not records:
            result = {"h2h_win_rate": np.nan, "h2h_draw_rate": np.nan, "h2h_avg_goal_diff": np.nan}
            self._h2h_cache[key] = result
            return result

        n = len(records)
        result = {
            "h2h_win_rate": sum(1 for r in records if r["result"] == "a_win") / n,
            "h2h_draw_rate": sum(1 for r in records if r["result"] == "draw") / n,
            "h2h_avg_goal_diff": sum(r["gd"] for r in records) / n,
        }
        self._h2h_cache[key] = result
        return result

    # ── WC player aggregates ──────────────────────────────────────────────────

    def _build_global_name_initials(self):
        """Frequency-based team_name -> initials mapping from WC players data."""
        if self.players_df.empty or "MatchID" not in self.matches_df.columns:
            return
        match_lookup = (
            self.matches_df.set_index("MatchID")[["Home Team Name", "Away Team Name"]]
            .to_dict("index")
        )
        counts: dict[tuple, int] = {}
        for match_id, group in self.players_df.groupby("MatchID"):
            if match_id not in match_lookup:
                continue
            home = match_lookup[match_id]["Home Team Name"]
            away = match_lookup[match_id]["Away Team Name"]
            for initials in group["Team Initials"].unique():
                for name in [home, away]:
                    key = (name, initials)
                    counts[key] = counts.get(key, 0) + 1

        name_best: dict[str, dict] = {}
        for (name, initials), cnt in counts.items():
            name_best.setdefault(name, {})[initials] = cnt

        self._global_name_initials = {
            name: max(d, key=lambda k: d[k]) for name, d in name_best.items()
        }

    def _player_aggregates(self, team_initials: str | None, year: int) -> float:
        key = (team_initials, year)
        if key in self._player_agg_cache:
            return self._player_agg_cache[key]

        if not team_initials or self.players_df.empty:
            self._player_agg_cache[key] = 0.0
            return 0.0
        prior_ids = set(self.matches_df.loc[self.matches_df["Year"] < year, "MatchID"])
        if not prior_ids:
            self._player_agg_cache[key] = 0.0
            return 0.0
        mask = (
            self.players_df["MatchID"].isin(prior_ids) &
            (self.players_df["Team Initials"] == team_initials)
        )
        tp = self.players_df[mask]
        if tp.empty:
            self._player_agg_cache[key] = 0.0
            return 0.0
        distinct = tp["Player Name"].nunique()
        if distinct == 0:
            self._player_agg_cache[key] = 0.0
            return 0.0
        goal_events = tp["Event"].dropna().str.contains("G", na=False).sum()
        result = float(goal_events) / float(distinct)
        self._player_agg_cache[key] = result
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode_stage(self, stage_str: str) -> int:
        return STAGE_ORDINAL_MAP.get(stage_str, 1)

    def _fill_cold_start(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN rolling stats with global mean for teams with < 3 prior appearances."""
        stat_cols = [
            "team_win_rate", "team_draw_rate", "team_loss_rate",
            "team_avg_goals_scored", "team_avg_goals_conceded", "team_avg_goal_diff",
            "opp_win_rate", "opp_draw_rate", "opp_loss_rate",
            "opp_avg_goals_scored", "opp_avg_goals_conceded", "opp_avg_goal_diff",
        ]
        col_means = {c: df[c].mean() for c in stat_cols if c in df.columns}

        def _appearances(team: str, year: int) -> int:
            return int(self.results_df.loc[
                ((self.results_df["home_team"] == team) | (self.results_df["away_team"] == team)) &
                (self.results_df["year"] < year),
                "year"
            ].nunique())

        result = df.copy()
        for idx, row in result.iterrows():
            if _appearances(row["focal_team"], row["year"]) < 3:
                for col in stat_cols:
                    if col in result.columns and pd.isna(result.at[idx, col]):
                        result.at[idx, col] = col_means.get(col, 0.0)
        return result

    @staticmethod
    def _wc_to_results(matches_df: pd.DataFrame) -> pd.DataFrame:
        """Convert WC matches to results_df format as fallback."""
        # Clean the dataframe first
        matches_df = matches_df.dropna(subset=["Year", "Home Team Name", "Away Team Name", 
                                                "Home Team Goals", "Away Team Goals"]).copy()
        matches_df["Year"] = matches_df["Year"].astype(int)
        
        rows = []
        for _, r in matches_df.iterrows():
            rows.append({
                "date": r.get("Datetime", pd.NaT),
                "year": int(r["Year"]),
                "home_team": r["Home Team Name"],
                "away_team": r["Away Team Name"],
                "home_score": int(r["Home Team Goals"]),
                "away_score": int(r["Away Team Goals"]),
                "tournament": "FIFA World Cup",
                "neutral": True,
            })
        return pd.DataFrame(rows)

    # ── Main build ────────────────────────────────────────────────────────────

    def build_features(self) -> pd.DataFrame:
        """Build feature DataFrame from WC matches with symmetric augmentation."""
        rows = []

        for _, match in self.matches_df.iterrows():
            year_raw = match["Year"]
            if pd.isna(year_raw):
                continue
            year = int(year_raw)
            home_team = match["Home Team Name"]
            away_team = match["Away Team Name"]
            home_goals = match["Home Team Goals"]
            away_goals = match["Away Team Goals"]
            if pd.isna(home_goals) or pd.isna(away_goals):
                continue
            home_goals = int(home_goals)
            away_goals = int(away_goals)
            stage = match.get("Stage", "Group Stage")
            dt = match.get("Datetime", pd.Timestamp(f"{year}-06-01"))
            if pd.isna(dt):
                dt = pd.Timestamp(f"{year}-06-01")

            stage_ordinal = self._encode_stage(stage)
            t_weight = 1.0  # WC matches always weight 1.0
            is_neutral = 1  # WC matches are always at neutral venues

            # ELO before this match
            home_elo = self.get_elo_before(home_team, dt)
            away_elo = self.get_elo_before(away_team, dt)

            # Rolling stats
            hs = self._rolling_team_stats(home_team, year)
            as_ = self._rolling_team_stats(away_team, year)

            # H2H
            h2h_h = self._head_to_head_stats(home_team, away_team, year)
            h2h_a = self._head_to_head_stats(away_team, home_team, year)

            # Player aggregates
            h_init = self._global_name_initials.get(home_team)
            a_init = self._global_name_initials.get(away_team)
            h_goals_pp = self._player_aggregates(h_init, year)
            a_goals_pp = self._player_aggregates(a_init, year)

            # Outcome labels
            if home_goals > away_goals:
                out_h, out_a = "Home Win", "Away Win"
            elif home_goals < away_goals:
                out_h, out_a = "Away Win", "Home Win"
            else:
                out_h = out_a = "Draw"

            def _row(focal_stats, opp_stats, h2h, focal_elo, opp_elo,
                     focal_gpp, opp_gpp, focal_team, opp_team, outcome):
                return {
                    "focal_team": focal_team,
                    "opponent": opp_team,
                    "year": year,
                    "team_win_rate": focal_stats["win_rate"],
                    "team_draw_rate": focal_stats["draw_rate"],
                    "team_loss_rate": focal_stats["loss_rate"],
                    "team_avg_goals_scored": focal_stats["avg_goals_scored"],
                    "team_avg_goals_conceded": focal_stats["avg_goals_conceded"],
                    "team_avg_goal_diff": focal_stats["avg_goal_diff"],
                    "opp_win_rate": opp_stats["win_rate"],
                    "opp_draw_rate": opp_stats["draw_rate"],
                    "opp_loss_rate": opp_stats["loss_rate"],
                    "opp_avg_goals_scored": opp_stats["avg_goals_scored"],
                    "opp_avg_goals_conceded": opp_stats["avg_goals_conceded"],
                    "opp_avg_goal_diff": opp_stats["avg_goal_diff"],
                    "h2h_team_win_rate": h2h["h2h_win_rate"],
                    "h2h_draw_rate": h2h["h2h_draw_rate"],
                    "h2h_avg_goal_diff": h2h["h2h_avg_goal_diff"],
                    "team_elo": focal_elo,
                    "opp_elo": opp_elo,
                    "elo_diff": focal_elo - opp_elo,
                    "stage_ordinal": stage_ordinal,
                    "tournament_weight": t_weight,
                    "is_neutral": is_neutral,
                    "team_avg_goal_events_per_player": focal_gpp,
                    "opp_avg_goal_events_per_player": opp_gpp,
                    TARGET_COL: outcome,
                }

            rows.append(_row(hs, as_, h2h_h, home_elo, away_elo, h_goals_pp, a_goals_pp,
                             home_team, away_team, out_h))
            rows.append(_row(as_, hs, h2h_a, away_elo, home_elo, a_goals_pp, h_goals_pp,
                             away_team, home_team, out_a))

        df = pd.DataFrame(rows)
        df = self._fill_cold_start(df)
        return df

    def build_features_for_match(
        self,
        home_team: str,
        away_team: str,
        stage_ordinal: int = 6,
        is_neutral: bool = True,
        tournament_weight: float = 1.0,
    ) -> pd.DataFrame:
        """Build a single-row feature vector for prediction (no target column)."""
        future_year = int(self.results_df["year"].max()) + 1
        future_date = pd.Timestamp(f"{future_year}-01-01")

        hs = self._rolling_team_stats(home_team, future_year)
        as_ = self._rolling_team_stats(away_team, future_year)
        h2h = self._head_to_head_stats(home_team, away_team, future_year)

        home_elo = CURRENT_ELO_RATINGS.get(
            home_team, self._final_elo.get(home_team, ELO_INITIAL_RATING)
        )
        away_elo = CURRENT_ELO_RATINGS.get(
            away_team, self._final_elo.get(away_team, ELO_INITIAL_RATING)
        )

        h_init = self._global_name_initials.get(home_team)
        a_init = self._global_name_initials.get(away_team)
        h_gpp = self._player_aggregates(h_init, future_year)
        a_gpp = self._player_aggregates(a_init, future_year)

        row = {
            "team_win_rate": hs["win_rate"],
            "team_draw_rate": hs["draw_rate"],
            "team_loss_rate": hs["loss_rate"],
            "team_avg_goals_scored": hs["avg_goals_scored"],
            "team_avg_goals_conceded": hs["avg_goals_conceded"],
            "team_avg_goal_diff": hs["avg_goal_diff"],
            "opp_win_rate": as_["win_rate"],
            "opp_draw_rate": as_["draw_rate"],
            "opp_loss_rate": as_["loss_rate"],
            "opp_avg_goals_scored": as_["avg_goals_scored"],
            "opp_avg_goals_conceded": as_["avg_goals_conceded"],
            "opp_avg_goal_diff": as_["avg_goal_diff"],
            "h2h_team_win_rate": h2h["h2h_win_rate"],
            "h2h_draw_rate": h2h["h2h_draw_rate"],
            "h2h_avg_goal_diff": h2h["h2h_avg_goal_diff"],
            "team_elo": home_elo,
            "opp_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "stage_ordinal": stage_ordinal,
            "tournament_weight": tournament_weight,
            "is_neutral": int(is_neutral),
            "team_avg_goal_events_per_player": h_gpp,
            "opp_avg_goal_events_per_player": a_gpp,
            "year": future_year,
        }
        return pd.DataFrame([row], columns=FEATURE_COLS)
