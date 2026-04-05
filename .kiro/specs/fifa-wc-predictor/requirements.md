# Requirements Document

## Introduction

A machine learning system that predicts FIFA World Cup match outcomes and simulates full tournament brackets. The system ingests historical World Cup match data (1930–present), engineers team-level features, trains a gradient boosting classifier to predict head-to-head match results, and uses those predictions to simulate a complete 32-team tournament and surface the most likely winner.

## Glossary

- **Predictor**: The end-to-end ML system described in this document
- **Match_Outcome_Model**: The trained gradient boosting classifier (XGBoost or LightGBM) that predicts the result of a single match between two teams
- **Feature_Engineer**: The component responsible for deriving team-level features from raw CSV data
- **Tournament_Simulator**: The component that runs a full World Cup bracket simulation using the Match_Outcome_Model
- **Raw_Data**: The three source CSV files — WorldCupMatches.csv, WorldCupPlayers.csv, WorldCups.csv
- **Team_Feature_Vector**: A numeric representation of a team's historical performance used as model input
- **Match_Record**: A single row representing one historical match with home team, away team, goals, stage, and outcome
- **Outcome**: One of three discrete labels — Home Win, Away Win, or Draw
- **Stage**: The round of the tournament (Group Stage, Round of 16, Quarter-final, Semi-final, Final)
- **Simulation_Run**: One complete execution of the tournament bracket from group stage through final
- **Win_Probability**: A value in [0, 1] representing the model's confidence that a given team wins a match

## Requirements

### Requirement 1: Data Ingestion and Validation

**User Story:** As a data scientist, I want the system to load and validate the raw CSV files, so that downstream processing operates on clean, well-structured data.

#### Acceptance Criteria

1. THE Feature_Engineer SHALL load WorldCupMatches.csv, WorldCupPlayers.csv, and WorldCups.csv from a configurable data directory path.
2. WHEN a required CSV file is missing from the data directory, THE Feature_Engineer SHALL raise a descriptive error identifying the missing file.
3. WHEN a CSV file contains rows with null values in required columns (Year, Home Team Name, Away Team Name, Home Team Goals, Away Team Goals), THE Feature_Engineer SHALL log a warning and exclude those rows from processing.
4. THE Feature_Engineer SHALL parse the Datetime column into a standardised date format and expose the parsed year as a numeric feature.
5. WHEN duplicate MatchID values are detected in WorldCupMatches.csv, THE Feature_Engineer SHALL deduplicate by retaining the first occurrence and logging the count of removed duplicates.

---

### Requirement 2: Feature Engineering

**User Story:** As a data scientist, I want meaningful team-level features derived from historical match data, so that the model has signal beyond raw scores.

#### Acceptance Criteria

1. THE Feature_Engineer SHALL compute the following per-team rolling statistics over all prior World Cup matches: win rate, draw rate, loss rate, average goals scored, average goals conceded, and average goal difference.
2. THE Feature_Engineer SHALL compute head-to-head win rate, draw rate, and average goal difference between each pair of teams across all historical matches.
3. THE Feature_Engineer SHALL encode the match Stage as an ordinal numeric feature (Group Stage = 1, Round of 16 = 2, Quarter-final = 3, Semi-final = 4, Third Place = 5, Final = 6).
4. WHEN computing rolling statistics for a match in year Y, THE Feature_Engineer SHALL use only matches from years strictly before Y to prevent data leakage.
5. THE Feature_Engineer SHALL produce a symmetric training record for each match — one row with Team A as home and one row with Team A as away — so the model learns team identity independent of home/away assignment.
6. THE Feature_Engineer SHALL derive a player-level aggregate feature per team per tournament: average number of goal events per player from WorldCupPlayers.csv.
7. WHEN a team has fewer than 3 prior World Cup appearances, THE Feature_Engineer SHALL fill missing rolling statistics with the global mean of that statistic across all teams.

---

### Requirement 3: Match Outcome Model Training

**User Story:** As a data scientist, I want a trained classifier that predicts match outcomes, so that I can evaluate its accuracy and use it for simulation.

#### Acceptance Criteria

1. THE Match_Outcome_Model SHALL be trained to predict Outcome (Home Win, Away Win, Draw) as a three-class classification problem using the Team_Feature_Vectors produced by the Feature_Engineer.
2. THE Match_Outcome_Model SHALL use a gradient boosting algorithm (XGBoost or LightGBM) as the base estimator.
3. THE Predictor SHALL perform a temporal train/test split: all matches before a configurable cutoff year form the training set, and all matches from the cutoff year onward form the test set.
4. WHEN training is complete, THE Predictor SHALL report classification metrics on the test set: accuracy, precision, recall, and F1-score per class.
5. THE Match_Outcome_Model SHALL output a Win_Probability for each of the three outcome classes for any given match input.
6. THE Predictor SHALL perform hyperparameter tuning via cross-validation on the training set and select the configuration that maximises weighted F1-score.
7. THE Predictor SHALL serialize the trained Match_Outcome_Model to a file so that it can be reloaded without retraining.
8. WHEN the serialized model file is loaded, THE Predictor SHALL produce identical Win_Probability outputs as the originally trained model for the same inputs (round-trip property).

---

### Requirement 4: Head-to-Head Match Prediction

**User Story:** As a user, I want to query the predicted outcome of any two-team matchup, so that I can explore individual match predictions.

#### Acceptance Criteria

1. WHEN a user provides two valid team names, THE Predictor SHALL return the Win_Probability for each of the three Outcomes (Home Win, Away Win, Draw).
2. WHEN a user provides two valid team names, THE Predictor SHALL return the predicted most-likely Outcome label.
3. IF either team name is not present in the historical dataset, THEN THE Predictor SHALL return a descriptive error identifying the unrecognised team name.
4. THE Predictor SHALL accept team names in a case-insensitive manner.
5. FOR ALL valid team pairs (A, B), the sum of Win_Probability values across the three Outcome classes SHALL equal 1.0 (within floating-point tolerance of 1e-6).

---

### Requirement 5: Tournament Bracket Simulation

**User Story:** As a user, I want to simulate a full World Cup tournament, so that I can see which team the model predicts as the most likely winner.

#### Acceptance Criteria

1. THE Tournament_Simulator SHALL accept a list of 32 participating team names and a group stage assignment as input.
2. THE Tournament_Simulator SHALL simulate the group stage by running all intra-group matches and advancing the top 2 teams per group based on predicted Win_Probability.
3. WHEN simulating a knockout match, THE Tournament_Simulator SHALL determine the winner by sampling from the Win_Probability distribution output by the Match_Outcome_Model, treating a Draw as requiring a penalty shootout resolved by a 50/50 coin flip.
4. THE Tournament_Simulator SHALL run a configurable number of Simulation_Runs (default 1000) and aggregate results to produce a tournament win probability for each team.
5. THE Tournament_Simulator SHALL output a ranked list of all 32 teams sorted by their tournament win probability in descending order.
6. WHEN the number of Simulation_Runs is set to 1, THE Tournament_Simulator SHALL return a single deterministic bracket result.
7. FOR ALL valid inputs, the sum of tournament win probabilities across all 32 teams SHALL equal 1.0 (within floating-point tolerance of 1e-4).

---

### Requirement 6: Results Reporting

**User Story:** As a user, I want clear visualisations and a summary report of predictions, so that I can interpret and share the model's findings.

#### Acceptance Criteria

1. THE Predictor SHALL produce a bar chart showing the top 10 teams ranked by tournament win probability.
2. THE Predictor SHALL produce a confusion matrix visualisation of the Match_Outcome_Model's test-set predictions.
3. THE Predictor SHALL produce a feature importance chart showing the top 20 features by model-assigned importance score.
4. WHEN generating charts, THE Predictor SHALL save each chart as a PNG file to a configurable output directory.
5. THE Predictor SHALL produce a summary CSV file containing each team's tournament win probability, semi-final probability, and final probability aggregated across all Simulation_Runs.
