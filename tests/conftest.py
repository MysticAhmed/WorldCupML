"""Shared pytest fixtures for FIFA World Cup Predictor tests."""

import os
import pytest
import pandas as pd


@pytest.fixture
def matches_df():
    """Sample WorldCupMatches DataFrame with 10+ rows spanning multiple years and teams."""
    data = {
        "MatchID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "Year": [1930, 1930, 1934, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974],
        "Datetime": [
            "13 Jul 1930 - 15:00",
            "14 Jul 1930 - 15:00",
            "27 May 1934 - 16:30",
            "27 May 1934 - 16:30",
            "04 Jun 1938 - 17:00",
            "24 Jun 1950 - 15:00",
            "16 Jun 1954 - 17:00",
            "08 Jun 1958 - 15:00",
            "30 May 1962 - 14:00",
            "11 Jul 1966 - 15:00",
            "21 Jun 1970 - 16:00",
            "07 Jul 1974 - 16:00",
        ],
        "Stage": [
            "Group 1",
            "Group 1",
            "Group 1",
            "Group 2",
            "Quarter-finals",
            "Group 1",
            "Group 1",
            "Semi-finals",
            "Quarter-finals",
            "Final",
            "Semi-finals",
            "Final",
        ],
        "Home Team Name": [
            "France", "USA", "Germany", "Brazil", "France",
            "Brazil", "Germany", "France", "Brazil", "England",
            "Brazil", "Germany",
        ],
        "Away Team Name": [
            "Mexico", "Belgium", "Italy", "Spain", "Belgium",
            "Mexico", "Hungary", "Sweden", "England", "Germany",
            "Uruguay", "Netherlands",
        ],
        "Home Team Goals": [4, 3, 1, 3, 3, 4, 3, 5, 3, 4, 3, 2],
        "Away Team Goals": [1, 0, 2, 0, 1, 0, 8, 2, 1, 2, 1, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def players_df():
    """Sample WorldCupPlayers DataFrame."""
    data = {
        "MatchID": [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        "Team Initials": ["FRA", "FRA", "MEX", "USA", "BEL", "GER", "ITA", "BRA", "ESP", "FRA"],
        "Player Name": [
            "Player A", "Player B", "Player C",
            "Player D", "Player E",
            "Player F", "Player G",
            "Player H", "Player I",
            "Player J",
        ],
        "Event": ["G40", "G65", None, "G20", None, "G10", "G55", "G30", None, "G70"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def cups_df():
    """Sample WorldCups DataFrame."""
    data = {
        "Year": [1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974],
        "Winner": [
            "Uruguay", "Italy", "Italy", "Uruguay", "Germany FR",
            "Brazil", "Brazil", "England", "Brazil", "Germany FR",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def tmp_data_dir(tmp_path, matches_df, players_df, cups_df):
    """Write sample DataFrames to a temporary directory and return the path."""
    matches_df.to_csv(os.path.join(tmp_path, "WorldCupMatches.csv"), index=False)
    players_df.to_csv(os.path.join(tmp_path, "WorldCupPlayers.csv"), index=False)
    cups_df.to_csv(os.path.join(tmp_path, "WorldCups.csv"), index=False)
    # Minimal results.csv so DataLoader doesn't raise FileNotFoundError
    results_data = {
        "date": ["1930-07-13", "1930-07-14", "1934-05-27", "1938-06-04",
                 "1950-06-24", "1954-06-16", "1958-06-08", "1962-05-30",
                 "1966-07-11", "1970-06-21", "1974-07-07"],
        "home_team": ["France", "USA", "Germany", "France",
                      "Brazil", "Germany", "France", "Brazil",
                      "England", "Brazil", "Germany"],
        "away_team": ["Mexico", "Belgium", "Italy", "Belgium",
                      "Mexico", "Hungary", "Sweden", "England",
                      "Germany", "Uruguay", "Netherlands"],
        "home_score": [4, 3, 1, 3, 4, 3, 5, 3, 4, 3, 2],
        "away_score": [1, 0, 2, 1, 0, 8, 2, 1, 2, 1, 1],
        "tournament": ["FIFA World Cup"] * 11,
        "city": ["Montevideo"] * 11,
        "country": ["Uruguay"] * 11,
        "neutral": ["TRUE"] * 11,
    }
    import pandas as pd
    pd.DataFrame(results_data).to_csv(os.path.join(tmp_path, "results.csv"), index=False)
    return str(tmp_path)
