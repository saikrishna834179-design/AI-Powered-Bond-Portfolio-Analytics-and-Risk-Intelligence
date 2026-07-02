# ===============================
# AI Bond Price Prediction
# Step 1 - Import Libraries
# ===============================

import pandas as pd
import numpy as np

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Model Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Visualization
import matplotlib.pyplot as plt

# ===============================
# Load Dataset
# ===============================

df = pd.read_csv("bond_portfolio_data.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display missing values
print("\nMissing Values:")
print(df.isnull().sum())

# ===============================
# Step 2 - Data Preprocessing
# ===============================

# Select features (X)
features = [
    'CouponRate',
    'YearsToMaturity',
    'YieldToMaturity',
    'ModifiedDuration',
    'MacaulayDuration',
    'EffectiveDuration',
    'Convexity',
    'EffectiveConvexity',
    'DV01_Per100Face',
    'OAS_bps',
    'ZSpread_bps',
    'SpreadOverBenchmark_bps',
    'CreditRating',
    'Currency',
    'BondType',
    'Sector'
]

# Target variable (Y)
target = 'CleanPrice'

# Create X and y
X = df[features]
y = df[target]

# Convert categorical columns into numbers
X = pd.get_dummies(
    X,
    columns=['CreditRating', 'Currency', 'BondType', 'Sector'],
    drop_first=True
)

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)

# Split data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data :", X_test.shape)

# ===============================
# Step 3 - Linear Regression Model
# ===============================

# Create the model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Predict on test data
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_r2 = r2_score(y_test, y_pred_lr)

print("\n========== Linear Regression Results ==========")
print(f"MAE  : {lr_mae:.4f}")
print(f"RMSE : {lr_rmse:.4f}")
print(f"R²   : {lr_r2:.4f}")

# ===============================
# Step 4 - Random Forest Model
# ===============================

# Create the model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)

print("\n========== Random Forest Results ==========")
print(f"MAE  : {rf_mae:.4f}")
print(f"RMSE : {rf_rmse:.4f}")
print(f"R²   : {rf_r2:.4f}")

# ===============================
# Step 5 - Save Results for Power BI
# ===============================

# Predictions (Linear Regression)
predictions = pd.DataFrame({
    'BondID': df.loc[X_test.index, 'BondID'],
    'ActualPrice': y_test.values,
    'PredictedPrice': y_pred_lr,
    'PredictionError': y_test.values - y_pred_lr,
    'Model': 'Linear Regression'
})

predictions.to_csv("Predictions.csv", index=False)

# Model Performance
metrics = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MAE': [lr_mae, rf_mae],
    'RMSE': [lr_rmse, rf_rmse],
    'R2': [lr_r2, rf_r2]
})

metrics.to_csv("ModelMetrics.csv", index=False)

# Feature Importance (Random Forest)
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

importance.to_csv("FeatureImportance.csv", index=False)

print("\n====================================")
print("CSV files created successfully!")
print("1. Predictions.csv")
print("2. ModelMetrics.csv")
print("3. FeatureImportance.csv")
print("====================================")