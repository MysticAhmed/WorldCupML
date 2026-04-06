"""Test the model's accuracy on 2018 and 2022 World Cup matches."""

import pandas as pd
import joblib
from predictor.config import FEATURE_COLS, OUTCOME_INT_MAP
from predictor.data_loader import DataLoader
from predictor.feature_engineer import FeatureEngineer

# Load the trained model
model = joblib.load('models/match_outcome_model.joblib')

# Load and prepare data
loader = DataLoader('.')
matches_df, players_df, cups_df, results_df = loader.load()

# Create features
engineer = FeatureEngineer(matches_df, players_df, results_df)
features_df = engineer.build_features()

# Filter for 2018 and 2022 World Cup matches
test_2018 = features_df[features_df['year'] == 2018]
test_2022 = features_df[features_df['year'] == 2022]
test_all = features_df[features_df['year'].isin([2018, 2022])]

def evaluate(df, label):
    """Evaluate model on a dataset."""
    if len(df) == 0:
        print(f"\n{label}: No data found")
        return
    
    X = df[list(FEATURE_COLS)]
    y_true = df['outcome'].tolist()
    y_pred_int = model.predict(X)
    y_pred = [OUTCOME_INT_MAP[i] for i in y_pred_int]
    
    # Calculate accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)
    
    # Count outcomes
    from collections import Counter
    outcome_counts = Counter(y_true)
    
    print(f"\n{label}")
    print(f"  Total matches: {len(df)}")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{len(y_true)})")
    print(f"  Actual outcomes: {dict(outcome_counts)}")
    
    # Show some predictions
    print(f"\n  Sample predictions:")
    cols = df.columns.tolist()
    home_col = 'home' if 'home' in cols else 'Home Team Name'
    away_col = 'away' if 'away' in cols else 'Away Team Name'
    
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        home = row.get(home_col, row.get('home', 'Unknown'))
        away = row.get(away_col, row.get('away', 'Unknown'))
        print(f"    {home} vs {away}: Predicted={y_pred[i]}, Actual={y_true[i]}")

print("=" * 70)
print("Model Evaluation on Recent World Cups")
print("=" * 70)

evaluate(test_2018, "2018 World Cup")
evaluate(test_2022, "2022 World Cup")
evaluate(test_all, "Combined 2018 + 2022")

print("\n" + "=" * 70)
