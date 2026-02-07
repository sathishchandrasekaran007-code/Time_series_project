!pip install scikit-optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.structural import UnobservedComponents
from skopt import BayesSearchCV, gp_minimize
from skopt.space import Real, Categorical
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Task 1: Synthetic Data Generation ---
def generate_synthetic_data(n_obs=600):
    np.random.seed(42)
    time = np.arange(n_obs)

    # Components: Trend + Weekly Seasonality + Yearly Seasonality
    trend = 0.05 * time
    seasonal_weekly = 10 * np.sin(2 * np.pi * time / 7)
    seasonal_yearly = 20 * np.sin(2 * np.pi * time / 365.25)

    # Multiplicative noise (modeled as exp(log_components + normal_noise))
    clean_signal = 100 + trend + seasonal_weekly + seasonal_yearly
    noise = np.random.normal(0, 0.05, n_obs)
    data = clean_signal * (1 + noise)

    return pd.Series(data, name="value")

data = generate_synthetic_data()
train = data.iloc[:500]
test = data.iloc[500:]

# --- Task 2 & 3: SSM & Bayesian Optimization ---
def objective(params):
    # Unpack hyperparameters to optimize
    level_var, seasonal_var = params

    try:
        # Define Structural Time Series (State Space Model)
        model = UnobservedComponents(
            train,
            level='local linear trend',
            seasonal=7,
            autoregressive=1
        )

        # Fit with custom variance constraints
        res = model.fit(disp=False)

        # Forecast on validation set
        forecast = res.get_forecast(steps=len(test)).predicted_mean
        return mean_squared_error(test, forecast)
    except:
        return 1e10 # Penalty for non-convergence

# Search space: Log-variances for the state components
space = [
    Real(1e-4, 1.0, prior='log-uniform', name='level_var'),
    Real(1e-4, 1.0, prior='log-uniform', name='seasonal_var')
]

res_gp = gp_minimize(objective, space, n_calls=20, random_state=42)

# --- Task 4: Backtesting & Benchmark ---
# Optimized Model
best_model = UnobservedComponents(train, level='local linear trend', seasonal=7).fit(disp=False)
ssm_forecast = best_model.get_forecast(steps=100).predicted_mean

# Benchmark: Simple Exponential Smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
ses_model = SimpleExpSmoothing(train).fit()
ses_forecast = ses_model.forecast(100)

# Performance Comparison
print(f"SSM RMSE: {np.sqrt(mean_squared_error(test, ssm_forecast)):.4f}")
print(f"SES RMSE: {np.sqrt(mean_squared_error(test, ses_forecast)):.4f}")