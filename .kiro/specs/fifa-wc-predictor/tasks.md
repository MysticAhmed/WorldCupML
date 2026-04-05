# Implementation Plan: FIFA World Cup Predictor

## Overview

Implement the `predictor/` Python package module by module, following the pipeline order: config → data_loader → feature_engineer → model_trainer → predictor_api → simulator → reporter. Wire everything together in `worldCup.ipynb`. Tests (pytest + Hypothesis) are added as optional sub-tasks alongside each module.

## Tasks

- [x] 1. Set up project structure and shared config
  - Create `predictor/__init__.py` and `predictor/config.py` with shared constants (column names, stage ordinal map, outcome label map, default paths, random seed)
  - Create `tests/__init__.py` and `tests/conftest.py` with shared fixtures: sample `matches_df`, `players_df`, `cups_df` DataFrames and a `tmp_data_dir` fixture that writes them to disk
  - Add `requirements.txt` (or `pyproject.toml`) listing: pandas, scikit-learn, xgboost, lightgbm, joblib, hypothesis, pytest, matplotlib, seaborn
  - _Requirements: 1.1, 3.2_

- [x] 2. Implement DataLoader
  - [x] 2.1 Implement `DataLoader` class in `predictor/data_loader.py`
    - `__init__(self, data_dir: str)` stores path
    - `load()` reads all three CSVs, validates required columns, drops null rows with WARNING log, deduplicates MatchID keeping first with INFO log, parses Datetime and extracts numeric `Year`, returns `(matches_df, players_df, cups_df)`
    - Raise `FileNotFoundError` with filename in message for any missing file
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 2.2 Write property test: missing file raises descriptive error
    - **Property 1: Missing file raises descriptive error**
    - **Validates: Requirements 1.2**
    - Test function: `test_missing_file_raises_error` in `tests/test_data_loader.py`
    - Use `@given` + `@settings(max_examples=100)`; parametrise over each of the three filenames

  - [ ]* 2.3 Write property test: null rows excluded from output
    - **Property 2: Null rows are excluded from output**
    - **Validates: Requirements 1.3**
    - Test function: `test_null_rows_excluded` in `tests/test_data_loader.py`

  - [ ]* 2.4 Write property test: deduplication retains first occurrence
    - **Property 3: Deduplication retains first occurrence**
    - **Validates: Requirements 1.5**
    - Test function: `test_deduplication_retains_first` in `tests/test_data_loader.py`

  - [ ]* 2.5 Write property test: datetime parsing yields correct year
    - **Property 4: Datetime parsing yields correct year**
    - **Validates: Requirements 1.4**
    - Test function: `test_datetime_parsing_year` in `tests/test_data_loader.py`

  - [ ]* 2.6 Write unit test: DataLoader loads all three files successfully
    - Test function: `test_dataloader_loads_all_files` in `tests/test_data_loader.py`
    - Uses `tmp_data_dir` fixture; asserts all three DataFrames are non-empty
    - _Requirements: 1.1_

- [x] 3. Checkpoint — ensure DataLoader tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement FeatureEngineer
  - [x] 4.1 Implement `FeatureEngineer` class in `predictor/feature_engineer.py`
    - `__init__(self, matches_df, players_df)` stores DataFrames
    - `_rolling_team_stats(year)` — win/draw/loss rate, avg goals scored/conceded, avg goal diff using only `Year < year`
    - `_head_to_head_stats(team_a, team_b, year)` — h2h win rate, draw rate, avg goal diff before `year`
    - `_player_aggregates(team, year)` — avg goal events per player from `players_df` before `year`
    - `_encode_stage(stage_str)` — ordinal encoding per `config.py` map
    - `_fill_cold_start(df)` — replace NaN rolling stats with global column mean for teams with < 3 appearances
    - `build_features()` — symmetric augmentation (two rows per match), assembles full `Team_Feature_Vector` DataFrame
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [ ]* 4.2 Write property test: rolling stats use only prior-year data
    - **Property 5: Rolling stats use only prior-year data (leakage prevention)**
    - **Validates: Requirements 2.1, 2.4**
    - Test function: `test_rolling_stats_no_leakage` in `tests/test_feature_engineer.py`

  - [ ]* 4.3 Write property test: head-to-head stats use only prior-year data
    - **Property 6: Head-to-head stats use only prior-year data**
    - **Validates: Requirements 2.2, 2.4**
    - Test function: `test_h2h_stats_no_leakage` in `tests/test_feature_engineer.py`

  - [ ]* 4.4 Write property test: symmetric augmentation produces exactly two rows per match
    - **Property 7: Symmetric augmentation produces exactly two rows per match**
    - **Validates: Requirements 2.5**
    - Test function: `test_symmetric_augmentation` in `tests/test_feature_engineer.py`

  - [ ]* 4.5 Write property test: cold-start teams have no NaN rolling stats
    - **Property 8: Cold-start teams have no NaN rolling stats**
    - **Validates: Requirements 2.7**
    - Test function: `test_cold_start_no_nan` in `tests/test_feature_engineer.py`

  - [ ]* 4.6 Write property test: player aggregate matches manual calculation
    - **Property 9: Player aggregate matches manual calculation**
    - **Validates: Requirements 2.6**
    - Test function: `test_player_aggregate_correctness` in `tests/test_feature_engineer.py`

  - [ ]* 4.7 Write unit test: stage ordinal encoding maps all known stage strings
    - Test function: `test_stage_ordinal_encoding` in `tests/test_feature_engineer.py`
    - Assert each stage string in `config.py` maps to its expected integer
    - _Requirements: 2.3_

- [x] 5. Checkpoint — ensure FeatureEngineer tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [-] 6. Implement ModelTrainer
  - [x] 6.1 Implement `ModelTrainer` class in `predictor/model_trainer.py`
    - `__init__(self, features_df, cutoff_year, model_path)` stores args; raise `ValueError` if cutoff year is outside the data's year range
    - `train()` — temporal split (`Year < cutoff_year` → train, else → test); raise `ValueError` if either split is empty; hyperparameter search via `GridSearchCV`/`RandomizedSearchCV` maximising weighted F1; fit pipeline (preprocessor + XGBoost/LightGBM); return `TrainResult` dataclass with fitted estimator, metrics dict (`accuracy`, `precision`, `recall`, `f1`), and feature importance series
    - `save()` — `joblib.dump` full pipeline to `<model_path>/match_outcome_model.joblib`
    - `load(model_path)` — `joblib.load`; raise `FileNotFoundError` if file absent
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [ ] 6.2 Write property test: temporal split integrity
    - **Property 10: Temporal split integrity**
    - **Validates: Requirements 3.3**
    - Test function: `test_temporal_split_integrity` in `tests/test_model_trainer.py`

  - [ ] 6.3 Write property test: model output is a valid probability distribution
    - **Property 11: Model output is a valid probability distribution**
    - **Validates: Requirements 3.5**
    - Test function: `test_model_output_probability_simplex` in `tests/test_model_trainer.py`

  - [ ] 6.4 Write property test: model serialisation round-trip
    - **Property 12: Model serialisation round-trip**
    - **Validates: Requirements 3.7, 3.8**
    - Test function: `test_model_serialisation_round_trip` in `tests/test_model_trainer.py`

  - [ ] 6.5 Write unit tests for ModelTrainer
    - `test_model_has_three_classes` — assert `model.classes_` has length 3 (Req 3.1, 3.2)
    - `test_metrics_keys` — assert metrics dict contains `accuracy`, `precision`, `recall`, `f1` (Req 3.4)
    - _Requirements: 3.1, 3.2, 3.4_

- [x] 7. Checkpoint — ensure ModelTrainer tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [-] 8. Implement PredictorAPI
  - [x] 8.1 Implement `PredictorAPI` class in `predictor/predictor_api.py`
    - `__init__(self, model, feature_engineer)` stores args; build case-insensitive team name lookup from `feature_engineer`
    - `predict(home_team, away_team)` — normalise team names (case-insensitive); raise `ValueError` with team name in message for unknown names; build feature vector; call `model.predict_proba()`; return `MatchPrediction` dataclass with `home_win_prob`, `away_win_prob`, `draw_prob`, `predicted_label`
    - `MatchPrediction` dataclass defined in same file
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ] 8.2 Write property test: API probability sum equals 1.0
    - **Property 13: API probability sum equals 1.0**
    - **Validates: Requirements 4.1, 4.5**
    - Test function: `test_api_probability_sum` in `tests/test_predictor_api.py`

  - [ ] 8.3 Write property test: predicted label equals argmax of probabilities
    - **Property 14: Predicted label equals argmax of probabilities**
    - **Validates: Requirements 4.2**
    - Test function: `test_predicted_label_is_argmax` in `tests/test_predictor_api.py`

  - [ ] 8.4 Write property test: unknown team name raises ValueError
    - **Property 15: Unknown team name raises ValueError**
    - **Validates: Requirements 4.3**
    - Test function: `test_unknown_team_raises_value_error` in `tests/test_predictor_api.py`

  - [ ] 8.5 Write property test: case-insensitive team name lookup
    - **Property 16: Case-insensitive team name lookup**
    - **Validates: Requirements 4.4**
    - Test function: `test_case_insensitive_lookup` in `tests/test_predictor_api.py`

- [x] 9. Checkpoint — ensure PredictorAPI tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [-] 10. Implement TournamentSimulator
  - [x] 10.1 Implement `TournamentSimulator` class in `predictor/simulator.py`
    - `__init__(self, predictor, n_runs=1000, seed=42)` — raise `ValueError` if `n_runs < 1`; raise `ValueError` for unknown team names before simulation starts
    - `simulate(teams, groups)` — run `n_runs` full brackets: group stage (all intra-group matches, advance top 2 per group by win probability), knockout rounds (sample winner from probability distribution; Draw → 50/50 coin flip); aggregate per-team win/semifinal/final counts; return `SimulationResult` dataclass
    - `SimulationResult` exposes `win_probabilities`, `semifinal_probabilities`, `final_probabilities`, `ranked_teams` (sorted descending by win probability)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [ ] 10.2 Write property test: group stage advances exactly 2 teams per group
    - **Property 17: Group stage advances exactly 2 teams per group**
    - **Validates: Requirements 5.2**
    - Test function: `test_group_stage_advances_two_per_group` in `tests/test_simulator.py`

  - [ ] 10.3 Write property test: tournament win probabilities sum to 1.0
    - **Property 18: Tournament win probabilities sum to 1.0**
    - **Validates: Requirements 5.4, 5.7**
    - Test function: `test_tournament_win_probs_sum_to_one` in `tests/test_simulator.py`

  - [ ] 10.4 Write property test: ranked output is sorted descending and complete
    - **Property 19: Ranked output is sorted descending and complete**
    - **Validates: Requirements 5.5**
    - Test function: `test_ranked_output_sorted_and_complete` in `tests/test_simulator.py`

  - [ ] 10.5 Write unit test: TournamentSimulator accepts 32-team input without error
    - Test function: `test_simulator_accepts_32_teams` in `tests/test_simulator.py`
    - _Requirements: 5.1_

- [x] 11. Checkpoint — ensure TournamentSimulator tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [-] 12. Implement ReportGenerator
  - [x] 12.1 Implement `ReportGenerator` class in `predictor/reporter.py`
    - `__init__(self, output_dir)` — `os.makedirs(output_dir, exist_ok=True)`
    - `bar_chart(sim_result)` — top-10 teams by win probability, save as `win_probability_bar_chart.png`
    - `confusion_matrix(y_true, y_pred)` — save as `confusion_matrix.png`
    - `feature_importance(importance_series)` — top-20 features, save as `feature_importance.png`
    - `summary_csv(sim_result)` — write CSV with columns `team`, `win_prob`, `semifinal_prob`, `final_prob`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ] 12.2 Write property test: charts are saved as PNG files to the output directory
    - **Property 20: Charts are saved as PNG files to the output directory**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
    - Test function: `test_charts_saved_as_png` in `tests/test_reporter.py`

  - [ ] 12.3 Write property test: summary CSV contains required columns and valid probabilities
    - **Property 21: Summary CSV contains required columns and valid probabilities**
    - **Validates: Requirements 6.5**
    - Test function: `test_summary_csv_columns_and_probs` in `tests/test_reporter.py`

- [x] 13. Wire pipeline in `worldCup.ipynb`
  - [x] 13.1 Add notebook cells that instantiate and run the full pipeline in order
    - Cell 1: imports and config
    - Cell 2: `DataLoader` → load CSVs
    - Cell 3: `FeatureEngineer` → `build_features()`
    - Cell 4: `ModelTrainer` → `train()` → `save()`
    - Cell 5: `PredictorAPI` → example head-to-head query
    - Cell 6: `TournamentSimulator` → `simulate()` with 2026 WC teams and groups
    - Cell 7: `ReportGenerator` → all charts + summary CSV
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1_

- [x] 14. Final checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at each pipeline stage
- Property tests use `@settings(max_examples=100)` and are tagged with `# Feature: fifa-wc-predictor, Property N: <text>`
- Unit tests and property tests are complementary — both are needed for full coverage
