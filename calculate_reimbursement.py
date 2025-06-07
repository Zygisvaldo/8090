import sys
import pandas as pd
import numpy as np
import pickle

# Load the trained model
try:
    with open('reimbursement_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: reimbursement_model.pkl not found. Run train_model.py first.")
    sys.exit(1)

# Get command-line arguments
if len(sys.argv) != 4:
    print("Usage: python3 calculate_reimbursement.py <days> <miles> <receipts>")
    sys.exit(1)

try:
    days = float(sys.argv[1])
    miles = float(sys.argv[2])
    receipts = float(sys.argv[3])
except ValueError:
    print("Error: All inputs must be numbers")
    sys.exit(1)

# Create DataFrame with one case
data = [{
    'days': days,
    'miles': miles,
    'receipts': receipts
}]
df = pd.DataFrame(data)

# Feature engineering (same as train_model.py)
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

# Select features
X = df[['days', 'miles', 'receipts', 'miles_per_day', 'receipts_per_day', 
        'log_receipts', 'is_long_trip', 'cents', 'is_sweet_spot', 
        'high_receipt_penalty', 'days_receipts_interaction', 'miles_receipts_ratio']]

# Predict
prediction = model.predict(X)[0]
print(f"Predicted reimbursement: {prediction:.2f}")
