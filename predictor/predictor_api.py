"""PredictorAPI for the FIFA World Cup Predictor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from predictor.config import FEATURE_COLS, OUTCOME_INT_MAP, OUTCOME_LABEL_MAP
from predictor.feature_engineer import FeatureEngineer


@dataclass
class MatchPrediction:
    """Prediction result for a single match."""

    home_win_prob: float
    away_win_prob: float
    draw_prob: float
    predicted_label: str


class PredictorAPI:
    """Thin wrapper around the trained model for interactive match queries."""

    def __init__(self, model, feature_engineer: FeatureEngineer):
        self.model = model
        self.feature_engineer = feature_engineer

        # Build case-insensitive team name lookup from WC matches + international results
        all_teams: set[str] = set()
        df = feature_engineer.matches_df
        all_teams.update(df["Home Team Name"].dropna().unique())
        all_teams.update(df["Away Team Name"].dropna().unique())
        if hasattr(feature_engineer, "results_df") and not feature_engineer.results_df.empty:
            rdf = feature_engineer.results_df
            all_teams.update(rdf["home_team"].dropna().unique())
            all_teams.update(rdf["away_team"].dropna().unique())
        self._team_lookup: dict[str, str] = {t.lower(): t for t in all_teams}

    def _resolve_team(self, name: str) -> str:
        """Return canonical team name or raise ValueError if not found."""
        canonical = self._team_lookup.get(name.lower())
        if canonical is None:
            raise ValueError(f"Unrecognised team name: '{name}'")
        return canonical

    def predict(self, home_team: str, away_team: str) -> MatchPrediction:
        """Predict the outcome probabilities for a matchup."""
        home_canonical = self._resolve_team(home_team)
        away_canonical = self._resolve_team(away_team)

        X = self.feature_engineer.build_features_for_match(
            home_canonical, away_canonical, stage_ordinal=6, is_neutral=True
        )

        proba = self.model.predict_proba(X)[0]

        home_win_prob = float(proba[OUTCOME_LABEL_MAP["Home Win"]])
        draw_prob = float(proba[OUTCOME_LABEL_MAP["Draw"]])
        away_win_prob = float(proba[OUTCOME_LABEL_MAP["Away Win"]])

        best_class_int = int(np.argmax(proba))
        predicted_label = OUTCOME_INT_MAP[best_class_int]

        return MatchPrediction(
            home_win_prob=home_win_prob,
            away_win_prob=away_win_prob,
            draw_prob=draw_prob,
            predicted_label=predicted_label,
        )
