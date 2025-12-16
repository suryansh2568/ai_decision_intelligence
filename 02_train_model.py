import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'retail_daily_aggregated.csv')
model_path = os.path.join(script_dir, '..', 'data', 'demand_model_v2.pkl')

print("Loading data...")
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['SKU_ID', 'Date']) # Sort by SKU then Date for Lag features

# Advanced Feature

# A. Lag Features
df['Lag_1'] = df.groupby('SKU_ID')['Quantity_Sold'].shift(1)
df['Lag_7'] = df.groupby('SKU_ID')['Quantity_Sold'].shift(7)

# B. Rolling Features
df['Rolling_Mean_7'] = df.groupby('SKU_ID')['Quantity_Sold'].transform(lambda x: x.rolling(window=7).mean())

# C. Date Features 
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter

# Drop NaNs created by lags (First 7 days will be empty)
df = df.dropna()

# Log Transformation
df['Log_Quantity'] = np.log1p(df['Quantity_Sold'])

# Preparing Data (One Hot Encoding)
df_encoded = pd.get_dummies(df, columns=['SKU_ID'], prefix='SKU')

features = [
    'Price', 'Competitor_Price', 
    'Lag_1', 'Lag_7', 'Rolling_Mean_7', 
    'DayOfWeek', 'Month', 'Is_Weekend'
] + [col for col in df_encoded.columns if 'SKU_' in col]

target = 'Log_Quantity' # Predicting Log Space

X = df_encoded[features]
y = df_encoded[target]

# Train/Test Split (80/20)
cutoff = int(len(df) * 0.8)
X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]

print(f"Training on {len(X_train)} rows. Testing on {len(X_test)} rows.")

# Train Model
print("Training XGBoost (with Log Target)...")
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01, # Slower learning to prevent overfitting
    max_depth=4,
    early_stopping_rounds=50,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)

# Evaluation
log_preds = model.predict(X_test)

# Converting back to Real Space
real_preds = np.expm1(log_preds)
real_actuals = np.expm1(y_test)

mae = mean_absolute_error(real_actuals, real_preds)
r2 = r2_score(real_actuals, real_preds)

print(f"\n--- Model Performance ---")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Save
joblib.dump({'model': model, 'features': features}, model_path)
print("Model v2 saved.")