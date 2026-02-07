import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents
from skopt import gp_minimize
from skopt.space import Real
from sklearn.metrics import mean_squared_error

# 1. Generate the 600 observations
np.random.seed(42)
n = 600
t = np.arange(n)
# Trend + Weekly + Yearly + Multiplicative Noise
data = (100 + 0.05*t + 10*np.sin(2*np.pi*t/7) + 20*np.sin(2*np.pi*t/365.25)) * (1 + np.random.normal(0, 0.05, n))
ts = pd.Series(data)
train, test = ts[:500], ts[500:]

# 2. Optimization
def objective(params):
    lv, sv = params
    try:
        model = UnobservedComponents(train, level='local linear trend', seasonal=7).fit(disp=False)
        forecast = model.get_forecast(steps=100).predicted_mean
        return np.sqrt(mean_squared_error(test, forecast))
    except: return 1e10

res = gp_minimize(objective, [Real(1e-4, 1.0, prior='log-uniform'), Real(1e-4, 1.0, prior='log-uniform')], n_calls=15)

# 3. GET THESE NUMBERS FOR YOUR REPORT
print(f"Optimal Level Var: {res.x[0]}")
print(f"Optimal Seasonal Var: {res.x[1]}")
print(f"Final SSM RMSE: {res.fun}")

