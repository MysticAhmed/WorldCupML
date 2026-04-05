"""DataLoader: loads WC CSVs + international results dataset."""

import logging
import os

import pandas as pd

from predictor.config import (
    CUPS_FILENAME, CUPS_REQUIRED_COLS,
    FORMER_NAMES_FILENAME,
    MATCHES_FILENAME, MATCHES_REQUIRED_COLS,
    PLAYERS_FILENAME, PLAYERS_REQUIRED_COLS,
    RESULTS_FILENAME, RESULTS_REQUIRED_COLS,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads all data sources and normalises team names via former_names.csv."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._name_map: dict[str, str] = {}  # former -> current

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> tuple:
        """Load and validate all data sources.

        Returns:
            (matches_df, players_df, cups_df, results_df)
            results_df contains all international matches from results.csv,
            with team names normalised to current names.

        Raises:
            FileNotFoundError: if WorldCupMatches, WorldCupPlayers, WorldCups,
                               or results.csv are missing.
        """
        # Build name normalisation map first (optional file)
        self._load_former_names()

        # Required WC files
        matches_df = self._read_csv(MATCHES_FILENAME, required=True)
        players_df = self._read_csv(PLAYERS_FILENAME, required=True)
        cups_df = self._read_csv(CUPS_FILENAME, required=True)

        # Required international results
        results_df = self._read_csv(RESULTS_FILENAME, required=True)

        # Validate & clean
        matches_df = self._drop_nulls(matches_df, MATCHES_REQUIRED_COLS, MATCHES_FILENAME)
        players_df = self._drop_nulls(players_df, PLAYERS_REQUIRED_COLS, PLAYERS_FILENAME)
        cups_df = self._drop_nulls(cups_df, CUPS_REQUIRED_COLS, CUPS_FILENAME)
        results_df = self._drop_nulls(results_df, RESULTS_REQUIRED_COLS, RESULTS_FILENAME)

        matches_df = self._deduplicate_match_id(matches_df)
        matches_df = self._parse_wc_datetime(matches_df)
        
        # Drop rows where Year is still NaN after datetime parsing
        matches_df = matches_df.dropna(subset=["Year"]).reset_index(drop=True)
        matches_df["Year"] = matches_df["Year"].astype(int)

        # Parse results.csv date and extract year
        results_df["date"] = pd.to_datetime(results_df["date"], errors="coerce")
        results_df["year"] = results_df["date"].dt.year
        results_df = results_df.dropna(subset=["year"])
        results_df["year"] = results_df["year"].astype(int)

        # Normalise team names in results_df
        results_df["home_team"] = results_df["home_team"].map(
            lambda x: self._name_map.get(x, x)
        )
        results_df["away_team"] = results_df["away_team"].map(
            lambda x: self._name_map.get(x, x)
        )

        # Also normalise WC team names
        matches_df["Home Team Name"] = matches_df["Home Team Name"].map(
            lambda x: self._name_map.get(x, x)
        )
        matches_df["Away Team Name"] = matches_df["Away Team Name"].map(
            lambda x: self._name_map.get(x, x)
        )

        # neutral column: coerce to bool
        if "neutral" in results_df.columns:
            results_df["neutral"] = results_df["neutral"].map(
                lambda x: str(x).strip().upper() == "TRUE"
            )
        else:
            results_df["neutral"] = False

        logger.info(
            "Loaded %d WC matches, %d international results.",
            len(matches_df), len(results_df),
        )

        return matches_df, players_df, cups_df, results_df

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_former_names(self):
        """Build former->current name map from former_names.csv (optional)."""
        path = os.path.join(self.data_dir, FORMER_NAMES_FILENAME)
        if not os.path.exists(path):
            logger.warning("former_names.csv not found; skipping name normalisation.")
            return
        df = pd.read_csv(path)
        if "current" in df.columns and "former" in df.columns:
            for _, row in df.iterrows():
                self._name_map[row["former"]] = row["current"]
        # Hard-code common WC-era aliases not covered by the file
        extra = {
            "Germany FR": "Germany",
            "West Germany": "Germany",
            "Soviet Union": "Russia",
            "Yugoslavia": "Serbia",
            "Czechoslovakia": "Czech Republic",
            "Dutch East Indies": "Indonesia",
            "Korea Republic": "South Korea",
            "IR Iran": "Iran",
            "United States": "USA",
        }
        self._name_map.update(extra)

    def _read_csv(self, filename: str, required: bool = True) -> pd.DataFrame:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Required CSV file not found: {filename}")
            return pd.DataFrame()
        return pd.read_csv(path)

    def _drop_nulls(self, df: pd.DataFrame, required_cols: list, filename: str) -> pd.DataFrame:
        cols_to_check = [c for c in required_cols if c in df.columns]
        mask = df[cols_to_check].isnull().any(axis=1)
        n_dropped = int(mask.sum())
        if n_dropped > 0:
            logger.warning("%s: dropping %d row(s) with nulls in required columns.", filename, n_dropped)
        return df[~mask].reset_index(drop=True)

    def _deduplicate_match_id(self, df: pd.DataFrame) -> pd.DataFrame:
        if "MatchID" not in df.columns:
            return df
        n_before = len(df)
        df = df.drop_duplicates(subset=["MatchID"], keep="first").reset_index(drop=True)
        n_dup = n_before - len(df)
        if n_dup > 0:
            logger.info("%s: removed %d duplicate MatchID rows.", MATCHES_FILENAME, n_dup)
        return df

    def _parse_wc_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Datetime" not in df.columns:
            return df
        df["Datetime"] = pd.to_datetime(
            df["Datetime"].str.strip(), format="%d %b %Y - %H:%M", errors="coerce"
        )
        # Extract year from Datetime, but keep original Year if Datetime parsing failed
        parsed_year = df["Datetime"].dt.year
        if "Year" in df.columns:
            # Use parsed year where available, otherwise keep original
            df["Year"] = parsed_year.fillna(df["Year"])
        else:
            df["Year"] = parsed_year
        return df
