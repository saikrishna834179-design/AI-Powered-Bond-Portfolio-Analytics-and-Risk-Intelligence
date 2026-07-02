# AI-Powered Bond Portfolio Analytics and Risk Intelligence

An end-to-end fixed-income analytics project built using Python, Machine Learning, Monte Carlo Simulation, and Power BI.

This project analyzes a bond portfolio using pricing, yield, duration, convexity, credit spreads, Value at Risk, stress testing, and machine learning-based bond price prediction.

---

## Project Overview

The system transforms bond portfolio data into interactive dashboards and predictive insights.

It helps analyze:

- Portfolio value and bond holdings
- Yield to maturity and bond pricing
- Duration, convexity, and DV01 risk
- Credit rating and sector exposure
- Yield curve behavior
- Monte Carlo simulation and Value at Risk
- Bond spread analytics
- Machine learning-based bond clean price prediction

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Power BI
- DAX
- Monte Carlo Simulation

---

## Project Dashboards

### 1. Portfolio Overview Dashboard
Provides a high-level view of portfolio value, total bonds, sector allocation, credit ratings, and maturity distribution.

### 2. Duration and Risk Analytics Dashboard
Analyzes modified duration, Macaulay duration, convexity, DV01, and interest-rate sensitivity.

### 3. Yield Curve Analysis Dashboard
Shows yield behavior across different maturity tenors and supports interest-rate trend analysis.

### 4. Monte Carlo Risk Simulation Dashboard
Simulates portfolio profit and loss under interest-rate scenarios and estimates Value at Risk.

### 5. Bond Pricing and Spread Analytics Dashboard
Analyzes clean price, dirty price, yield to maturity, OAS, Z-spread, and spread over benchmark.

### 6. AI-Powered Bond Price Prediction Dashboard
Compares actual versus predicted bond prices and evaluates machine learning model performance.

### 7. Executive Bond Risk Lab Dashboard
Combines portfolio, risk, yield curve, Value at Risk, sector exposure, and ML insights into one executive dashboard.

---

## Machine Learning Models

The project uses the following regression models for bond clean price prediction:

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 0.7490 | 1.2612 | 0.9598 |
| Random Forest Regression | 1.8938 | 3.7729 | 0.6398 |

### Best Model

**Linear Regression** was the best-performing model.

- MAE: 0.7490
- RMSE: 1.2612
- R² Score: 0.9598

The model explained approximately 96% of the variation in bond clean prices within the testing dataset.

---

## Key Findings

- Portfolio contains 300 bonds across sectors, ratings, maturities, and bond types.
- Average Modified Duration: 4.86 years.
- Average Yield to Maturity: 7.13%.
- Estimated Portfolio Value at Risk: approximately ₹3.4 million.
- Linear Regression achieved the best bond price prediction performance.
- Coupon Rate, Yield to Maturity, Duration, and Convexity were important features influencing bond price.

---

## Repository Files

```text
bond_portfolio_data.csv        - Main bond portfolio dataset
yield_curve_history.csv        - Yield curve historical data
monte_carlo_scenarios.csv      - Monte Carlo simulation scenarios
bond_prediction.py             - Machine learning bond price prediction code
Predictions.csv                - Actual vs predicted bond prices
ModelMetrics.csv               - ML model performance metrics
FeatureImportance.csv          - Feature importance results
AI-Powered-Bond-Portfolio-Analytics-and-Risk-Intelligence.pbix
                               - Power BI dashboard file
CONVEXITY SENSITIVITY AI report.pdf
                               - Project report
CONVEXITY SENSITIVITY AI DASHBOARDS.pdf
                               - Dashboard screenshots
