"""Shared constants and configuration for the FIFA World Cup Predictor."""

# Random seed for reproducibility
RANDOM_SEED = 42

# Default paths
DEFAULT_MODEL_PATH = "models/"
DEFAULT_OUTPUT_DIR = "outputs/"

# ── WC CSV filenames ──────────────────────────────────────────────────────────
MATCHES_FILENAME = "WorldCupMatches.csv"
PLAYERS_FILENAME = "WorldCupPlayers.csv"
CUPS_FILENAME = "WorldCups.csv"

# ── International dataset filenames ──────────────────────────────────────────
RESULTS_FILENAME = "results.csv"
FORMER_NAMES_FILENAME = "former_names.csv"
GOALSCORERS_FILENAME = "goalscorers.csv"
SHOOTOUTS_FILENAME = "shootouts.csv"

# ── Required columns ─────────────────────────────────────────────────────────
MATCHES_REQUIRED_COLS = [
    "Year", "Home Team Name", "Away Team Name",
    "Home Team Goals", "Away Team Goals", "MatchID", "Stage", "Datetime",
]
PLAYERS_REQUIRED_COLS = ["MatchID", "Team Initials", "Player Name", "Event"]
CUPS_REQUIRED_COLS = ["Year", "Winner"]
RESULTS_REQUIRED_COLS = ["date", "home_team", "away_team", "home_score", "away_score"]

# ── Stage ordinal encoding ────────────────────────────────────────────────────
STAGE_ORDINAL_MAP = {
    "Group Stage": 1,
    "Group 1": 1, "Group 2": 1, "Group 3": 1, "Group 4": 1,
    "Group 5": 1, "Group 6": 1, "Group 7": 1, "Group 8": 1,
    "Group A": 1, "Group B": 1, "Group C": 1, "Group D": 1,
    "Group E": 1, "Group F": 1, "Group G": 1, "Group H": 1,
    "Group I": 1, "Group J": 1, "Group K": 1, "Group L": 1,
    "Preliminary round": 1, "First round": 1,
    "Round of 32": 2,
    "Round of 16": 3,
    "Quarter-finals": 4,
    "Semi-finals": 5,
    "Third place": 6, "Match for third place": 6,
    "Final": 7,
}

# ── Tournament importance weights (for ELO K-factor scaling) ─────────────────
# Higher = more important match
TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 1.0,
    "Confederations Cup": 0.85,
    "Copa América": 0.85,
    "UEFA Euro": 0.85,
    "AFC Asian Cup": 0.80,
    "Africa Cup of Nations": 0.80,
    "CONCACAF Gold Cup": 0.75,
    "FIFA World Cup qualification": 0.70,
    "UEFA Euro qualification": 0.65,
    "Copa América qualification": 0.65,
    "Friendly": 0.40,
}
DEFAULT_TOURNAMENT_WEIGHT = 0.55  # for unlisted tournaments

# ── ELO parameters ────────────────────────────────────────────────────────────
ELO_K_BASE = 32          # base K-factor
ELO_INITIAL_RATING = 1500
ELO_HOME_ADVANTAGE = 100  # added to home team rating for neutral=False matches

# ── Current ELO ratings (April 2026, from eloratings.net) ────────────────────
# Used to seed the 2026 WC simulation
CURRENT_ELO_RATINGS = {
    "Spain": 2165, "Argentina": 2113, "France": 2082, "England": 2020,
    "Brazil": 1984, "Portugal": 1984, "Colombia": 1975, "Netherlands": 1961,
    "Ecuador": 1933, "Croatia": 1930, "Germany": 1923, "Norway": 1912,
    "Japan": 1904, "Turkey": 1902, "Uruguay": 1892, "Switzerland": 1889,
    "Senegal": 1879, "Denmark": 1870, "Belgium": 1866, "Mexico": 1858,
    "Italy": 1856, "Paraguay": 1833, "Austria": 1827, "Morocco": 1821,
    "Canada": 1784, "Australia": 1783, "Russia": 1776, "Serbia": 1769,
    "Scotland": 1767, "Ukraine": 1767, "Iran": 1760, "South Korea": 1752,
    "Nigeria": 1752, "Greece": 1752, "Algeria": 1743, "Panama": 1737,
    "Poland": 1729, "United States": 1721, "Sweden": 1719, "Chile": 1710,
    "Hungary": 1703, "Peru": 1695, "Egypt": 1689, "Ivory Coast": 1676,
    "Tunisia": 1636, "Cameroon": 1614, "Costa Rica": 1613, "Ghana": 1505,
    "Saudi Arabia": 1568, "Qatar": 1425, "Korea Republic": 1752,
    "USA": 1721,  # alias
}

# ── Outcome label encoding ────────────────────────────────────────────────────
OUTCOME_LABEL_MAP = {"Home Win": 0, "Draw": 1, "Away Win": 2}
OUTCOME_INT_MAP = {v: k for k, v in OUTCOME_LABEL_MAP.items()}

# ── Feature columns ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Rolling stats (all international matches)
    "team_win_rate",
    "team_draw_rate",
    "team_loss_rate",
    "team_avg_goals_scored",
    "team_avg_goals_conceded",
    "team_avg_goal_diff",
    "opp_win_rate",
    "opp_draw_rate",
    "opp_loss_rate",
    "opp_avg_goals_scored",
    "opp_avg_goals_conceded",
    "opp_avg_goal_diff",
    # Head-to-head
    "h2h_team_win_rate",
    "h2h_draw_rate",
    "h2h_avg_goal_diff",
    # ELO ratings
    "team_elo",
    "opp_elo",
    "elo_diff",
    # Match context
    "stage_ordinal",
    "tournament_weight",
    "is_neutral",
    # WC-specific player aggregates
    "team_avg_goal_events_per_player",
    "opp_avg_goal_events_per_player",
    "year",
]

TARGET_COL = "outcome"
