"""Break down model accuracy by stage: group stage vs knockout rounds."""

import pandas as pd
import joblib
from predictor.config import FEATURE_COLS, OUTCOME_INT_MAP, OUTCOME_LABEL_MAP
from predictor.data_loader import DataLoader
from predictor.feature_engineer import FeatureEngineer
from sklearn.metrics import accuracy_score

model = joblib.load('models/match_outcome_model.joblib')
loader = DataLoader('.')
matches_df, players_df, cups_df, results_df = loader.load()
engineer = FeatureEngineer(matches_df, players_df, results_df)
features_df = engineer.build_features()

# Test set only (unseen), one row per match
test_df = features_df[features_df['year'] >= 2018].iloc[::2].copy()

stage_names = {
    1: 'Group Stage', 3: 'Round of 16', 4: 'Quarter-finals',
    5: 'Semi-finals', 6: 'Third Place', 7: 'Final'
}

def eval_stage(df, label):
    if len(df) == 0:
        return
    X = df[list(FEATURE_COLS)]
    y_true = df['outcome'].map(OUTCOME_LABEL_MAP).values
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    correct = int(sum(y_true == y_pred))
    print(f'  {label:<20} {correct:>2}/{len(df):<3} = {acc:.1%}')

print('=' * 60)
print('ACCURACY BY STAGE — Test Set Only (2018 + 2022)')
print('=' * 60)

group = test_df[test_df['stage_ordinal'] == 1]
knockout = test_df[test_df['stage_ordinal'] > 1]

print('\nHigh-Level Split:')
eval_stage(group, 'GROUP STAGE')
eval_stage(knockout, 'KNOCKOUT (all)')

print('\nDetailed Breakdown:')
for ordinal in sorted(test_df['stage_ordinal'].unique()):
    stage_df = test_df[test_df['stage_ordinal'] == ordinal]
    name = stage_names.get(int(ordinal), f'Stage {int(ordinal)}')
    eval_stage(stage_df, name)

for year in [2018, 2022]:
    print(f'\n{"=" * 60}')
    print(f'{year} WORLD CUP BY STAGE')
    print('=' * 60)
    year_df = test_df[test_df['year'] == year]
    
    yg = year_df[year_df['stage_ordinal'] == 1]
    yk = year_df[year_df['stage_ordinal'] > 1]
    eval_stage(yg, 'GROUP STAGE')
    eval_stage(yk, 'KNOCKOUT (all)')
    print()
    for ordinal in sorted(year_df['stage_ordinal'].unique()):
        stage_df = year_df[year_df['stage_ordinal'] == ordinal]
        name = stage_names.get(int(ordinal), f'Stage {int(ordinal)}')
        eval_stage(stage_df, name)
