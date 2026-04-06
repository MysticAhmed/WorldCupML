https://www.kaggle.com/datasets/jahaidulislam/fifa-world-cup-1930-2022-all-match-dataset?
https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017?


Model used: It's an XGBoost classifier (XGBClassifier). Here's the full picture:

Model: XGBoost (Gradient Boosted Decision Trees)

Pipeline:

SimpleImputer - fills missing feature values with column means
XGBClassifier - the actual classifier, predicting one of 3 outcomes: Home Win, Draw, Away Win
Training setup:

Hyperparameter tuning via RandomizedSearchCV (40 iterations, 3-5 fold cross-validation)
Tuned params: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma
Uses balanced class weights to handle the fact that draws are less common than wins
Optimizes for weighted F1 score
Why XGBoost makes sense here: it handles tabular data well, is robust to missing values, captures non-linear relationships between features (like ELO differences, form, head-to-head), and is generally strong for sports prediction tasks.