import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Load data
data_path = 'data.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found")
data = pd.read_csv(data_path)


df = data[['tariff_rate', 'import_volume']]
y = data['price']

cv = KFold(n_splits=min(5, len(data)), shuffle=True, random_state=42)

# 1) Linear Regression
pipe_lr = Pipeline([
    ('scale', StandardScaler()),
    ('lr', LinearRegression())
])
r2_lr = cross_val_score(pipe_lr, df, y, cv=cv, scoring='r2')
mse_lr = -cross_val_score(pipe_lr, df, y, cv=cv, scoring='neg_mean_squared_error')
pipe_lr.fit(df, y)

# 2) Random Forest with manual grid search
param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
best_score = -np.inf
best_params = {}
for n in param_grid['n_estimators']:
    for d in param_grid['max_depth']:
        rf = RandomForestRegressor(n_estimators=n, max_depth=d, random_state=42)
        score = cross_val_score(rf, df, y, cv=cv, scoring='r2').mean()
        if score > best_score:
            best_score = score
            best_params = {'n_estimators': n, 'max_depth': d}
best_rf = RandomForestRegressor(**best_params, random_state=42)
best_rf.fit(df, y)
r2_full = best_rf.score(df, y)
mse_full = ((best_rf.predict(df) - y) ** 2).mean()


with open('model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)
metrics = {
    'Linear Regression': {
        'CV R2': r2_lr.round(3).tolist(),
        'CV MSE': mse_lr.round(3).tolist()
    },
    'Random Forest': {
        'Best Params': best_params,
        'CV R2': round(best_score, 3),
        'Full R2': round(r2_full, 3),
        'Full MSE': round(mse_full, 3)
    }
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("Training complete. Metrics saved to metrics.json and model.pkl")