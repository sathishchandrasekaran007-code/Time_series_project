# ================================
# Advanced Time Series Forecasting
# State Space + Bayesian Optimization
# ================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

from skopt import gp_minimize
from skopt.space import Real

np.random.seed(42)

# -------------------------------
# Synthetic data generation
# -------------------------------

n = 600
t = np.arange(n)

trend = 0.03 * t
weekly = 2 * np.sin(2 * np.pi * t / 7)
yearly = 4 * np.sin(2 * np.pi * t / 365)
noise = np.random.normal(0, 1 + 0.005 * t)

y = (10 + trend + weekly + yearly) * (1 + 0.05 * noise)

data = pd.Series(
    y,
    index=pd.date_range("2020-01-01", periods=n, freq="D")
)

# -------------------------------
# Train / validation split
# -------------------------------

train_size = int(len(data) * 0.8)
train = data.iloc[:train_size]
valid = data.iloc[train_size:]

# -------------------------------
# Bayesian optimization objective
# -------------------------------

def objective(params):

    level_var, trend_var, seas_var = params

    try:
        model = UnobservedComponents(
            train,
            level="local level",
            trend=True,
            seasonal=7
        )

        result = model.fit(
            start_params=[level_var, trend_var, seas_var],
            disp=False,
            maxiter=200
        )

        forecast = result.forecast(len(valid))
        rmse = np.sqrt(mean_squared_error(valid, forecast))
        return rmse

    except:
        return 1e6


space = [
    Real(1e-6, 1.0),
    Real(1e-6, 1.0),
    Real(1e-6, 1.0)
]

res = gp_minimize(
    objective,
    space,
    n_calls=20,
    random_state=42
)

best_params = res.x

# -------------------------------
# Final optimized SSM
# -------------------------------

final_model = UnobservedComponents(
    train,
    level="local level",
    trend=True,
    seasonal=7
)

final_result = final_model.fit(
    start_params=best_params,
    disp=False,
    maxiter=300
)

ssm_forecast = final_result.forecast(len(valid))
ssm_rmse = np.sqrt(mean_squared_error(valid, ssm_forecast))

# -------------------------------
# Benchmark model
# -------------------------------

hw_model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=7
).fit()

hw_forecast = hw_model.forecast(len(valid))
hw_rmse = np.sqrt(mean_squared_error(valid, hw_forecast))

# -------------------------------
# Rolling backtesting functions
# -------------------------------

def rolling_ssm(series, window=300, horizon=7):

    errors = []

    for i in range(0, len(series) - window - horizon, horizon):

        train_slice = series.iloc[i:i+window]
        test_slice = series.iloc[i+window:i+window+horizon]

        try:
            m = UnobservedComponents(
                train_slice,
                level="local level",
                trend=True,
                seasonal=7
            )

            r = m.fit(
                start_params=best_params,
                disp=False,
                maxiter=200
            )

            p = r.forecast(horizon)
            errors.append(
                np.sqrt(mean_squared_error(test_slice, p))
            )

        except:
            pass

    return np.mean(errors)


def rolling_hw(series, window=300, horizon=7):

    errors = []

    for i in range(0, len(series) - window - horizon, horizon):

        train_slice = series.iloc[i:i+window]
        test_slice = series.iloc[i+window:i+window+horizon]

        try:
            m = ExponentialSmoothing(
                train_slice,
                trend="add",
                seasonal="add",
                seasonal_periods=7
            ).fit()

            p = m.forecast(horizon)
            errors.append(
                np.sqrt(mean_squared_error(test_slice, p))
            )

        except:
            pass

    return np.mean(errors)


ssm_roll_rmse = rolling_ssm(data)
hw_roll_rmse = rolling_hw(data)

# -------------------------------
# Results table
# -------------------------------

summary = pd.DataFrame({
    "Model": ["Optimized State Space Model", "Exponential Smoothing"],
    "Holdout RMSE": [ssm_rmse, hw_rmse],
    "Rolling RMSE": [ssm_roll_rmse, hw_roll_rmse]
})

print("\nBest hyperparameters (level, trend, seasonal):")
print(best_params)

print("\nSummary:")
print(summary)

# -------------------------------
# Plots
# -------------------------------

plt.figure(figsize=(12,4))
plt.plot(train.index, train, label="Train")
plt.plot(valid.index, valid, label="Validation")
plt.plot(valid.index, ssm_forecast, label="Optimized SSM")
plt.plot(valid.index, hw_forecast, label="Benchmark")
plt.legend()
plt.title("Forecast comparison")
plt.show()

plt.figure(figsize=(8,4))
plt.plot(res.func_vals)
plt.title("Bayesian Optimization Convergence")
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.show()

