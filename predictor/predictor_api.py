"""PredictorAPI for the FIFA World Cup Predictor.

This module provides a simple interface for predicting individual match outcomes.
It wraps the trained model and feature engineer to handle:
- Team name resolution (case-insensitive, handles aliases)
- Feature vector construction for new matchups
- Probability prediction for all three outcomes (home win, draw, away win)
- Result caching to avoid redundant predictions
"""

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
    """Thin wrapper around the trained model for interactive match queries.
    
    Usage:
        api = PredictorAPI(model, feature_engineer)
        prediction = api.predict("Brazil", "Argentina")
        print(f"Brazil win: {prediction.home_win_prob:.1%}")
        print(f"Draw: {prediction.draw_prob:.1%}")
        print(f"Argentina win: {prediction.away_win_prob:.1%}")
    """

    def __init__(self, model, feature_engineer: FeatureEngineer):
        self.model = model
        self.feature_engineer = feature_engineer
        self._prediction_cache: dict[tuple[str, str], MatchPrediction] = {}

        # Build case-insensitive team name lookup from WC matches + international results
        # This allows users to type "brazil" instead of "Brazil"
        all_teams: set[str] = set()
        df = feature_engineer.matches_df
        all_teams.update(df["Home Team Name"].dropna().unique())
        all_teams.update(df["Away Team Name"].dropna().unique())
        if hasattr(feature_engineer, "results_df") and not feature_engineer.results_df.empty:
            rdf = feature_engineer.results_df
            all_teams.update(rdf["home_team"].dropna().unique())
            all_teams.update(rdf["away_team"].dropna().unique())
        
        # Map lowercase names to canonical names
        self._team_lookup: dict[str, str] = {t.lower(): t for t in all_teams}

    def _resolve_team(self, name: str) -> str:
        """Return canonical team name or raise ValueError if not found."""
        canonical = self._team_lookup.get(name.lower())
        if canonical is None:
            raise ValueError(f"Unrecognised team name: '{name}'")
        return canonical

    def predict(self, home_team: str, away_team: str) -> MatchPrediction:
        """Predict the outcome probabilities for a matchup.
        
        Args:
            home_team: Name of home team (case-insensitive)
            away_team: Name of away team (case-insensitive)
            
        Returns:
            MatchPrediction with probabilities for all three outcomes
            
        Raises:
            ValueError: If either team name is not recognized
        """
        # Resolve team names to canonical form (handles case and aliases)
        home_canonical = self._resolve_team(home_team)
        away_canonical = self._resolve_team(away_team)

        # Check cache to avoid redundant feature computation
        cache_key = (home_canonical, away_canonical)
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        # Build feature vector for this matchup
        # Uses current ELO ratings and historical stats
        X = self.feature_engineer.build_features_for_match(
            home_canonical, away_canonical, stage_ordinal=6, is_neutral=True
        )

        # Get probability distribution over 3 outcomes
        # proba[0] = P(home win), proba[1] = P(draw), proba[2] = P(away win)
        proba = self.model.predict_proba(X)[0]

        home_win_prob = float(proba[OUTCOME_LABEL_MAP["Home Win"]])
        draw_prob = float(proba[OUTCOME_LABEL_MAP["Draw"]])
        away_win_prob = float(proba[OUTCOME_LABEL_MAP["Away Win"]])

        # Predicted label is the outcome with highest probability
        best_class_int = int(np.argmax(proba))
        predicted_label = OUTCOME_INT_MAP[best_class_int]

        prediction = MatchPrediction(
            home_win_prob=home_win_prob,
            away_win_prob=away_win_prob,
            draw_prob=draw_prob,
            predicted_label=predicted_label,
        )
        
        # Cache result for future queries
        self._prediction_cache[cache_key] = prediction
        return prediction
