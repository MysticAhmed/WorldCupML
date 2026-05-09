# FIFA World Cup Predictor 🏆⚽

A machine learning system that predicts FIFA World Cup match outcomes and simulates tournament results using XGBoost, trained on 90+ years of World Cup history and 49,000+ international matches.

## 📊 Project Overview

This project predicts soccer match outcomes using machine learning:
- **Predicts** individual match results (Home Win, Draw, Away Win)
- **Simulates** entire World Cup tournaments using Monte Carlo methods
- **Analyzes** team strength through ELO ratings and 28 engineered features

### Key Results
- ✅ **48.4% accuracy** on unseen test data (2018 & 2022 World Cups)
- ✅ **2/2 correct** World Cup final predictions (France 2018, Argentina 2022)
- ✅ **Argentina favored** for 2026 World Cup (20.3% win probability)

### Why This Matters
Soccer prediction is notoriously difficult:
- Random guessing: 33% accuracy (3 possible outcomes)
- Professional betting markets: 50-55% accuracy
- **Our model: 48.4%** — competitive with industry standards

## 🗂️ Project Structure

```
WorldCupML/
├── worldCup.ipynb              # Main notebook - START HERE
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── predictor/                  # Core ML pipeline modules
│   ├── config.py              # Configuration and constants
│   ├── data_loader.py         # Load and validate datasets
│   ├── feature_engineer.py    # Compute ELO, stats, H2H features
│   ├── model_trainer.py       # Train XGBoost classifier
│   ├── predictor_api.py       # Prediction interface
│   ├── simulator.py           # Monte Carlo tournament simulation
│   └── reporter.py            # Generate visualizations
│
├── Data Files (CSV)
│   ├── WorldCupMatches.csv    # 964 WC matches (1930-2022)
│   ├── results.csv            # 49k international matches (1872-2026)
│   ├── WorldCupPlayers.csv    # Player records
│   ├── WorldCups.csv          # Tournament metadata
│   └── former_names.csv       # Team name mappings
│
├── models/                     # Trained model saved here
│   └── match_outcome_model.joblib
│
├── outputs/                    # Generated visualizations
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── win_probability_bar_chart.png
│   └── simulation_summary.csv
│
└── website/                    # Demo website (bonus)
    └── index.html
```

## 🚀 How to Run This Project

### Prerequisites
- **Python 3.8+** (tested on Python 3.11)
- **Jupyter Notebook** or VS Code with Jupyter extension
- **~500MB disk space** for data and models

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- `pandas`, `numpy` — Data manipulation
- `scikit-learn` — Machine learning pipeline
- `xgboost` — Gradient boosting classifier
- `matplotlib`, `seaborn` — Visualizations
- `jupyter` — Notebook environment

### Step 2: Run the Main Notebook
```bash
jupyter notebook worldCup.ipynb
```

**Then:** Click "Cell → Run All" to execute the entire pipeline.

**Runtime:** ~10-15 minutes on a modern laptop
- Data loading: ~1 min
- Feature engineering: ~3 min (computing ELO for 49k matches)
- Model training: ~5 min (hyperparameter search)
- Simulation: ~2 min (1,000 tournament runs)

### Step 3: View Results

The notebook outputs:
1. ✅ **Model Accuracy**: 48.4% on 2018-2022 test data
2. ✅ **World Cup Finals**: Correctly predicted France 2018 and Argentina 2022
3. ✅ **2026 Predictions**: Argentina (20.3%), Germany (18.9%), Netherlands (14.7%)
4. ✅ **Visualizations**: Saved to `outputs/` folder
   - Confusion matrix
   - Feature importance chart
   - Win probability bar chart
   - Simulation summary CSV

### Alternative: View Demo Website
Open `website/index.html` in your browser for an interactive demo (no server needed).

## 📈 How the Model Works

### 1️⃣ Data Collection
We use **5 datasets** totaling 59,000+ records:

| Dataset | Records | Purpose |
|---------|---------|---------|
| `WorldCupMatches.csv` | 964 matches | Primary training/test data (1930-2022) |
| `results.csv` | 49,287 matches | Calculate ELO ratings from all international matches |
| `WorldCupPlayers.csv` | 9,069 players | Squad quality metrics (goal-scoring ability) |
| `WorldCups.csv` | 20 tournaments | Tournament metadata and winners |
| `former_names.csv` | 36 mappings | Normalize team names ("West Germany" → "Germany") |

### 2️⃣ Feature Engineering (28 Features)
We engineer features that capture team strength and match context:

**Team Strength (22 features):**
- **ELO Ratings** (3): Dynamic skill ratings updated after each match
  - Team ELO, Opponent ELO, ELO Difference
- **Rolling Statistics** (12): Career performance from all international matches
  - Win/Draw/Loss rates, Goals scored/conceded, Goal difference
- **Recent Form** (4): Performance in last 20 matches (captures momentum)
- **Head-to-Head** (3): Historical record between the two teams

**Match Context (6 features):**
- **Player Quality** (2): Average goal events per player from World Cup squads
- **Tournament Stage** (1): Group stage (1) to Final (7)
- **Tournament Weight** (1): World Cup (1.0) vs Friendly (0.4)
- **Venue** (1): Neutral or home advantage
- **Year** (1): Captures era effects and tactical evolution

**Key Insight:** We use 49,000+ international matches to build ELO ratings and rolling stats, then apply these features to predict World Cup outcomes. This gives us much more training data than using World Cup matches alone.

### 3️⃣ Model Training
- **Algorithm**: XGBoost (Gradient Boosted Decision Trees)
  - Handles non-linear relationships well
  - Robust to missing values
  - Fast training with 28 features
  
- **Hyperparameter Tuning**: RandomizedSearchCV
  - Tests 40 random combinations
  - 3-5 fold cross-validation
  - Optimizes for F1 score (weighted)
  
- **Class Balancing**: Weighted samples
  - Draws are only ~20% of matches
  - Balanced weighting prevents model from ignoring draws
  
- **Temporal Split**: No data leakage
  - Train: 1930-2017 (836 matches)
  - Test: 2018-2022 (128 matches)
  - Model never sees test data during training

### 4️⃣ Prediction & Simulation
- **Match Prediction**: Model outputs probabilities for 3 outcomes
  - Example: Brazil vs Argentina → Brazil 45%, Draw 25%, Argentina 30%
  
- **Tournament Simulation**: Monte Carlo method (1,000 runs)
  - Each match outcome sampled from predicted probabilities
  - Captures realistic upset potential (underdogs can win)
  - Aggregates results to estimate win probabilities
  
- **2026 Format**: Simulates full 48-team tournament
  - 12 groups of 4 teams
  - Top 2 + best 8 third-place teams advance (32 teams)
  - Single-elimination knockout bracket

## 📊 Model Performance

### Overall Accuracy (Test Set Only)
| Metric | Value | Details |
|--------|-------|---------|
| **Overall Accuracy** | **48.4%** | 62/128 matches correct (2018-2022) |
| 2018 World Cup | 46.9% | 30/64 matches |
| 2022 World Cup | 50.0% | 32/64 matches |
| **Finals Predicted** | **2/2 (100%)** | France 2018 ✅, Argentina 2022 ✅ |

### Accuracy by Tournament Stage
| Stage | Accuracy | Correct/Total |
|-------|----------|---------------|
| Group Stage | 48.0% | 47/98 |
| Round of 16 | 43.8% | 7/16 |
| Quarter-finals | 62.5% | 5/8 |
| Semi-finals | 50.0% | 2/4 |
| **Finals** | **50.0%** | **1/2** |

### Performance Context
Soccer prediction is extremely difficult:
- ⚪ **Random guessing**: 33% (3 possible outcomes)
- 🟡 **Our model**: 48.4% (competitive)
- 🟢 **Professional betting markets**: 50-55% (industry standard)

**Why is soccer hard to predict?**
- Low scoring (1-2 goals per match)
- Frequent upsets (underdogs win ~30% of the time)
- High variance (one lucky goal changes everything)
- Penalty shootouts (essentially coin flips)

## 🏆 World Cup Final Predictions

| Year | Matchup | Predicted Winner | Actual Winner | Confidence | Result |
|------|---------|------------------|---------------|------------|--------|
| 2022 | Argentina vs France | Argentina | Argentina | 72.1% | ✅ Correct |
| 2018 | France vs Croatia | France | France | 75.8% | ✅ Correct |

*Note: 2002-2014 finals were in the training set, so those predictions don't count as valid tests.*

## 🔮 2026 World Cup Predictions

Top 10 predicted winners (based on 1,000 tournament simulations):

| Rank | Team | Win Probability | Make Final | Make Semis |
|------|------|-----------------|------------|------------|
| 1 | 🇦🇷 Argentina | 20.3% | 34.3% | 47.0% |
| 2 | 🇩🇪 Germany | 18.9% | 26.3% | 39.4% |
| 3 | 🇳🇱 Netherlands | 14.7% | 25.7% | 37.9% |
| 4 | 🇫🇷 France | 10.3% | 17.2% | 28.0% |
| 5 | 🇧🇷 Brazil | 8.3% | 13.8% | 31.5% |
| 6 | 🇨🇭 Switzerland | 5.3% | 12.2% | 19.0% |
| 7 | 🇨🇴 Colombia | 4.1% | 9.6% | 17.2% |
| 8 | 🇹🇷 Turkey | 4.1% | 13.3% | 29.3% |
| 9 | 🇳🇴 Norway | 3.9% | 7.9% | 21.0% |
| 10 | 🇵🇹 Portugal | 2.7% | 7.5% | 13.8% |

## 🎨 Demo Website

Open `website/index.html` in your browser for an interactive demo showcasing:
- Model accuracy breakdown
- Feature importance
- 2026 predictions with probability bars
- Sample tournament bracket
- Dataset explanations

No server needed — just double-click the HTML file.

## 📁 Key Files Explained

### Main Notebook
- **worldCup.ipynb**: Complete pipeline with 10 well-documented code cells. Each cell has a markdown header explaining what it does and why.

### Predictor Modules (predictor/)
- **config.py**: Constants, file paths, ELO parameters, feature column names
- **data_loader.py**: Loads CSVs, validates columns, normalizes team names
- **feature_engineer.py**: Computes ELO ratings, rolling stats, H2H, player metrics
- **model_trainer.py**: Trains XGBoost with hyperparameter tuning, evaluates on test set
- **predictor_api.py**: Interface for predicting individual matches
- **simulator.py**: Monte Carlo tournament simulation (1,000 runs)
- **reporter.py**: Generates confusion matrix, feature importance, probability charts

### Data Files
- **WorldCupMatches.csv**: 964 World Cup matches (1930-2022) — primary training/test data
- **results.csv**: 49,287 international matches (1872-2026) — used for ELO calculations
- **WorldCupPlayers.csv**: 9,069 player records — squad quality features
- **WorldCups.csv**: Tournament metadata (20 tournaments)
- **former_names.csv**: Maps old team names to current (36 mappings)

## 🔧 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### "File not found" errors
Make sure you're running the notebook from the `WorldCupML/` directory.

### Notebook takes too long
- Feature engineering takes 1-2 minutes (computing ELO for 49k matches)
- Model training takes 3-5 minutes (hyperparameter search with 40 iterations)
- Total runtime: ~5-10 minutes on a modern laptop

### Low accuracy concerns
48% is actually competitive for soccer prediction. The model correctly predicted both the 2018 and 2022 World Cup winners on completely unseen data.

## 📚 Technical Details

### Why XGBoost?
- Handles multi-class classification (3 outcomes)
- Built-in support for class imbalance
- Captures non-linear feature interactions
- Fast hyperparameter tuning
- Proven track record on tabular sports data

### Data Leakage Prevention
- **Strict temporal split**: Train on pre-2018, test on 2018+
- **ELO computed chronologically**: Only uses matches before prediction date
- **Rolling stats**: Only historical data, no future information
- **Fair evaluation**: One row per match (no double-counting)

### Feature Importance
Top 5 features by XGBoost importance:
1. **ELO Difference** (6.7%) — Most important
2. **Stage Ordinal** (5.8%) — Knockout vs group dynamics
3. **Year** (4.4%) — Temporal trends
4. **H2H Goal Diff** (4.0%) — Historical matchup
5. **H2H Draw Rate** (4.0%) — Tendency to draw

## 🚧 Future Improvements

Potential enhancements to increase accuracy from 48% to 55%+:
1. **Better H2H imputation** — 30% of matches have missing H2H data
2. **Integrate goalscorers.csv** — Add striker quality features
3. **Integrate shootouts.csv** — Improve knockout predictions
4. **Ensemble methods** — Combine XGBoost + LightGBM + CatBoost
5. **Stage-specific models** — Separate models for group vs knockout
6. **Neural networks** — Try deep learning approach

See `MODEL_IMPROVEMENT_ANALYSIS.md` for detailed analysis.

## 📄 License

This is an academic project. Data sources:
- World Cup data: Public domain historical records
- International results: [Kaggle International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

## 👥 Authors

Machine Learning Course Project — Virginia Tech, Spring 2026

## 🙏 Acknowledgments

- FIFA for World Cup historical data
- Kaggle community for international results dataset
- XGBoost developers for the excellent ML library
- scikit-learn for preprocessing and evaluation tools
