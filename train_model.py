import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import json

with open('public_cases.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame([{
    'days': case['input']['trip_duration_days'],
    'miles': case['input']['miles_traveled'],
    'receipts': case['input']['total_receipts_amount'],
    'reimbursement': case['expected_output'] 
} for case in data])

# Feature engineering
df['miles_per_day'] = df['miles'] / df['days']
df['receipts_per_day'] = df['receipts'] / df['days']
df['log_receipts'] = np.log1p(df['receipts'])
df['is_long_trip'] = (df['days'] >= 8).astype(int)
df['cents'] = df['receipts'] % 1
df['is_sweet_spot'] = ((df['miles_per_day'].between(180, 220)) & 
                       (df['receipts_per_day'].between(75, 120))).astype(int)
df['high_receipt_penalty'] = (df['receipts_per_day'] > 120).astype(int)
df['days_receipts_interaction'] = df['days'] * df['receipts_per_day']
df['miles_receipts_ratio'] = df['miles'] / (df['receipts'] + 1)

# Handle NaN/Inf
df.fillna(0, inplace=True)
df.replace([np.inf, -np.inf], 0, inplace=True)

# Features and target
X = df[['days', 'miles', 'receipts', 'miles_per_day', 'receipts_per_day', 
        'log_receipts', 'is_long_trip', 'cents', 'is_sweet_spot', 
        'high_receipt_penalty', 'days_receipts_interaction', 'miles_receipts_ratio']]
y = df['reimbursement']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', XGBRegressor(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [3, 6, 10],
    'rf__learning_rate': [0.01, 0.1]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_
print(f"Best params: {grid_search.best_params_}")
print(f"Test MAE: {-grid_search.score(X_test, y_test)}")

# Save model
with open('reimbursement_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Generate predictions for private_cases.json
with open('private_cases.json', 'r') as f:
    private_cases = json.load(f)
private_df = pd.DataFrame([{
    'days': case['trip_duration_days'],
    'miles': case['miles_traveled'],
    'receipts': case['total_receipts_amount']
} for case in private_cases])
private_df['miles_per_day'] = private_df['miles'] / private_df['days']
private_df['receipts_per_day'] = private_df['receipts'] / private_df['days']
private_df['log_receipts'] = np.log1p(private_df['receipts'])
private_df['is_long_trip'] = (private_df['days'] >= 8).astype(int)
private_df['cents'] = private_df['receipts'] % 1
private_df['is_sweet_spot'] = ((private_df['miles_per_day'].between(180, 220)) & 
                               (private_df['receipts_per_day'].between(75, 120))).astype(int)
private_df['high_receipt_penalty'] = (private_df['receipts_per_day'] > 120).astype(int)
private_df['days_receipts_interaction'] = private_df['days'] * private_df['receipts_per_day']
private_df['miles_receipts_ratio'] = private_df['miles'] / (private_df['receipts'] + 1)
private_df.fillna(0, inplace=True)
private_df.replace([np.inf, -np.inf], 0, inplace=True)
X_private = private_df[['days', 'miles', 'receipts', 'miles_per_day', 'receipts_per_day', 
                        'log_receipts', 'is_long_trip', 'cents', 'is_sweet_spot', 
                        'high_receipt_penalty', 'days_receipts_interaction', 'miles_receipts_ratio']]
predictions = model.predict(X_private)
np.savetxt('private_results.txt', predictions, fmt='%.2f')
