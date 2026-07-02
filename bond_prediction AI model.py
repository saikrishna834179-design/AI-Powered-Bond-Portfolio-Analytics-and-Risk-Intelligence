

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


df = pd.read_csv("bond_portfolio_data.csv")

print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

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

target = 'CleanPrice'
X = df[features]
y = df[target]
X = pd.get_dummies(
    X,
    columns=['CreditRating', 'Currency', 'BondType', 'Sector'],
    drop_first=True
)

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data :", X_test.shape)

lr_model = LinearRegression()


lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_r2 = r2_score(y_test, y_pred_lr)

print("\n========== Linear Regression Results ==========")
print(f"MAE  : {lr_mae:.4f}")
print(f"RMSE : {lr_rmse:.4f}")
print(f"R²   : {lr_r2:.4f}")

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)

print("\n========== Random Forest Results ==========")
print(f"MAE  : {rf_mae:.4f}")
print(f"RMSE : {rf_rmse:.4f}")
print(f"R²   : {rf_r2:.4f}")

predictions = pd.DataFrame({
    'BondID': df.loc[X_test.index, 'BondID'],
    'ActualPrice': y_test.values,
    'PredictedPrice': y_pred_lr,
    'PredictionError': y_test.values - y_pred_lr,
    'Model': 'Linear Regression'
})

predictions.to_csv("Predictions.csv", index=False)

metrics = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MAE': [lr_mae, rf_mae],
    'RMSE': [lr_rmse, rf_rmse],
    'R2': [lr_r2, rf_r2]
})

metrics.to_csv("ModelMetrics.csv", index=False)

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