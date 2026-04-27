"""Test if the model correctly predicts actual World Cup tournament winners."""

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

# Known World Cup winners
ACTUAL_WINNERS = {
    2018: "France",
    2022: "Argentina",
    2014: "Germany",
    2010: "Spain",
    2006: "Italy",
    2002: "Brazil",
}

print("=" * 80)
print("WORLD CUP WINNER PREDICTION ANALYSIS")
print("=" * 80)

for year, actual_winner in sorted(ACTUAL_WINNERS.items(), reverse=True):
    print(f"\n{'='*80}")
    print(f"{year} World Cup - Actual Winner: {actual_winner}")
    print(f"{'='*80}")
    
    # Get all matches from that tournament
    year_matches = features_df[features_df['year'] == year].copy()
    
    if len(year_matches) == 0:
        print(f"  ⚠️  No data available for {year}")
        continue
    
    # Get the final match
    final_matches = year_matches[year_matches['stage_ordinal'] == 7]  # Final = 7
    
    if len(final_matches) == 0:
        print(f"  ⚠️  No final match found for {year}")
        continue
    
    final = final_matches.iloc[0]
    home_team = final['focal_team']
    away_team = final['opponent']
    actual_outcome = final['outcome']
    
    print(f"\nFinal Match: {home_team} vs {away_team}")
    print(f"Actual Result: {actual_outcome}")
    
    # Predict the final
    X = final[list(FEATURE_COLS)].values.reshape(1, -1)
    prediction_int = model.predict(X)[0]
    prediction = OUTCOME_INT_MAP[prediction_int]
    
    # Get probabilities
    probabilities = model.predict_proba(X)[0]
    prob_dict = {OUTCOME_INT_MAP[i]: prob for i, prob in enumerate(probabilities)}
    
    print(f"\nModel Prediction: {prediction}")
    print(f"Probabilities:")
    print(f"  Home Win ({home_team}): {prob_dict['Home Win']:.1%}")
    print(f"  Draw: {prob_dict['Draw']:.1%}")
    print(f"  Away Win ({away_team}): {prob_dict['Away Win']:.1%}")
    
    # Determine who the model predicted as winner
    if prediction == "Home Win":
        predicted_winner = home_team
    elif prediction == "Away Win":
        predicted_winner = away_team
    else:
        # For draw, pick the team with higher probability after removing draw
        home_prob = prob_dict['Home Win']
        away_prob = prob_dict['Away Win']
        predicted_winner = home_team if home_prob > away_prob else away_team
        print(f"  (Draw predicted, but {predicted_winner} had higher win probability)")
    
    # Check if correct
    correct = predicted_winner == actual_winner
    
    print(f"\n{'✓' if correct else '✗'} Predicted Winner: {predicted_winner}")
    print(f"{'✓' if correct else '✗'} Actual Winner: {actual_winner}")
    
    if correct:
        print(f"  🎉 CORRECT!")
    else:
        print(f"  ❌ INCORRECT")
    
    # Show all knockout matches for this tournament
    print(f"\n--- Knockout Stage Accuracy for {year} ---")
    knockout = year_matches[year_matches['stage_ordinal'] >= 3]  # Round of 16 and beyond
    
    if len(knockout) > 0:
        X_knockout = knockout[list(FEATURE_COLS)]
        y_true = knockout['outcome'].tolist()
        y_pred_int = model.predict(X_knockout)
        y_pred = [OUTCOME_INT_MAP[i] for i in y_pred_int]
        
        correct_knockout = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy_knockout = correct_knockout / len(y_true)
        
        print(f"  Knockout matches: {len(knockout)}")
        print(f"  Correct predictions: {correct_knockout}/{len(knockout)}")
        print(f"  Accuracy: {accuracy_knockout:.1%}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Overall summary
correct_finals = 0
total_finals = 0

for year, actual_winner in ACTUAL_WINNERS.items():
    year_matches = features_df[features_df['year'] == year]
    if len(year_matches) == 0:
        continue
    
    final_matches = year_matches[year_matches['stage_ordinal'] == 7]
    if len(final_matches) == 0:
        continue
    
    total_finals += 1
    final = final_matches.iloc[0]
    home_team = final['focal_team']
    away_team = final['opponent']
    
    X = final[list(FEATURE_COLS)].values.reshape(1, -1)
    prediction_int = model.predict(X)[0]
    prediction = OUTCOME_INT_MAP[prediction_int]
    probabilities = model.predict_proba(X)[0]
    prob_dict = {OUTCOME_INT_MAP[i]: prob for i, prob in enumerate(probabilities)}
    
    if prediction == "Home Win":
        predicted_winner = home_team
    elif prediction == "Away Win":
        predicted_winner = away_team
    else:
        home_prob = prob_dict['Home Win']
        away_prob = prob_dict['Away Win']
        predicted_winner = home_team if home_prob > away_prob else away_team
    
    if predicted_winner == actual_winner:
        correct_finals += 1

print(f"\nFinal Match Predictions: {correct_finals}/{total_finals} correct ({correct_finals/total_finals*100:.1f}%)")
print(f"\nNote: The model was trained on data before 2018, so 2018 and 2022 are true test cases.")
print("=" * 80)
