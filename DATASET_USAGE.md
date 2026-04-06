# Dataset Usage Summary

This document explains how each dataset in the WorldCupML project is used by the machine learning model.

## Core Datasets (Used by Model)

### 1. **WorldCupMatches.csv** (4,700 matches)
- **Purpose**: Primary training data for match outcome prediction
- **Content**: All FIFA World Cup matches from 1930-2022
- **Usage**: 
  - Training/testing the match outcome classifier
  - Provides match results, scores, stages, venues, attendance
  - Split by CUTOFF_YEAR (2018): trains on 1930-2017, tests on 2018-2022
- **Size**: 261 KB
- **Key columns**: Year, Home/Away Team Names, Goals, Stage, Win conditions

### 2. **results.csv** (49,287 matches)
- **Purpose**: ELO rating calculation for all international teams
- **Content**: Complete international match history from 1872-2026
- **Usage**:
  - Calculates team strength (ELO ratings) over time
  - ELO features are critical predictors in the model
  - Includes friendlies, qualifiers, tournaments (not just World Cups)
- **Size**: 3,627 KB
- **Date range**: 1872 (first international match) to present
- **Key columns**: date, home_team, away_team, home_score, away_score, tournament, neutral

### 3. **WorldCupPlayers.csv** (37,784 player records)
- **Purpose**: Player-level statistics for team strength assessment
- **Content**: Individual player data from World Cup matches
- **Usage**:
  - Aggregates player quality metrics per team
  - Features like average goals per player, caps, etc.
  - Helps assess squad strength beyond just team-level stats
- **Size**: 2,100 KB
- **Key columns**: Player names, team initials, goals, position, events

### 4. **WorldCups.csv** (20 tournaments)
- **Purpose**: Tournament-level metadata
- **Content**: Summary info for each World Cup (1930-2014 in original)
- **Usage**:
  - Provides context about host countries, winners, attendance
  - Used for validation and reference
- **Size**: 1 KB
- **Key columns**: Year, Country, Winner, Runners-Up, Goals Scored, Attendance

### 5. **former_names.csv** (36 mappings)
- **Purpose**: Team name normalization
- **Content**: Maps historical team names to current names
- **Usage**:
  - Ensures consistency across datasets
  - Examples: "West Germany" → "Germany", "Soviet Union" → "Russia"
  - Critical for accurate ELO tracking and feature engineering
- **Size**: 2 KB
- **Key columns**: former, current

## Reference Datasets (Not Used by Model)

### 6. **goalscorers.csv** (47,601 goals)
- **Purpose**: Individual goal records
- **Content**: Every goal scored in international matches
- **Current Status**: NOT actively used by the model
- **Potential Use**: Could enhance player/team offensive features
- **Size**: 3,180 KB

### 7. **shootouts.csv** (675 penalty shootouts)
- **Purpose**: Penalty shootout results
- **Content**: Detailed shootout data from international matches
- **Current Status**: NOT actively used by the model
- **Potential Use**: Could improve knockout stage predictions
- **Size**: 28 KB

## Integration Datasets

### 8. **FIFA World Cup 1930-2022 All Match Dataset.csv** (964 matches)
- **Purpose**: Source dataset for 2018 and 2022 World Cup data
- **Content**: Complete World Cup match data in different format
- **Usage**:
  - Extracted 2018 (64 matches) and 2022 (64 matches) data
  - Converted to match existing WorldCupMatches.csv format
  - Integrated via `integrate_new_data.py` script
- **Size**: 216 KB
- **Status**: Integration complete, data now in WorldCupMatches.csv

### 9. **WorldCupMatches_backup.csv** (4,572 matches)
- **Purpose**: Backup of original data before 2018/2022 integration
- **Content**: WorldCupMatches.csv before adding recent tournaments
- **Usage**: Safety backup for rollback if needed
- **Size**: 246 KB

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LOADING                             │
├─────────────────────────────────────────────────────────────┤
│  DataLoader reads:                                           │
│  • WorldCupMatches.csv                                       │
│  • WorldCupPlayers.csv                                       │
│  • WorldCups.csv                                             │
│  • results.csv                                               │
│  • former_names.csv (for name normalization)                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                         │
├─────────────────────────────────────────────────────────────┤
│  FeatureEngineer creates:                                    │
│  • ELO ratings (from results.csv - 49k matches)              │
│  • Rolling team statistics (wins, goals, form)               │
│  • Head-to-head records                                      │
│  • Player aggregates (from WorldCupPlayers.csv)              │
│  • Home advantage indicators                                 │
│  • Tournament stage features                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                            │
├─────────────────────────────────────────────────────────────┤
│  ModelTrainer:                                               │
│  • Trains on matches before CUTOFF_YEAR (2018)               │
│  • Tests on 2018-2022 World Cup matches                      │
│  • Current accuracy: ~47% on test set                        │
│  • Saves model to models/match_outcome_model.joblib          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   PREDICTION & SIMULATION                    │
├─────────────────────────────────────────────────────────────┤
│  PredictorAPI + TournamentSimulator:                         │
│  • Predicts individual match outcomes                        │
│  • Runs Monte Carlo tournament simulations                   │
│  • Generates win probabilities for 2026 World Cup            │
└─────────────────────────────────────────────────────────────┘
```

## Model Performance

- **Training Data**: 1930-2017 World Cup matches (~850 matches)
- **Test Data**: 2018-2022 World Cup matches (128 matches)
- **Test Accuracy**: 46.88% (120/256 correct predictions)
- **Note**: ~47% accuracy is reasonable for soccer match prediction due to the sport's inherent unpredictability

## Key Features Used by Model

1. **ELO ratings** (most important) - from results.csv
2. **Recent form** - rolling win/loss/draw rates
3. **Head-to-head history** - past matchups between teams
4. **Player quality** - aggregated from WorldCupPlayers.csv
5. **Home advantage** - venue and neutral site indicators
6. **Tournament stage** - group vs knockout dynamics

## Future Enhancement Opportunities

- Integrate **goalscorers.csv** for offensive strength metrics
- Use **shootouts.csv** to improve knockout stage predictions
- Add more recent international results to **results.csv**
- Incorporate player-level data from recent tournaments
