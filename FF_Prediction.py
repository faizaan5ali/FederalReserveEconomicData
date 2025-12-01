"""
Predict the federal funds rate using FRB_H8.csv and constrain predicted quarterly
changes to ±0.75 points.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ----------------------------- LOAD DATA -----------------------------
fed = pd.read_csv("FEDFUNDS.csv")
h8 = pd.read_csv("FRB_H8.csv")

# ----------------------------- CLEAN H8 -----------------------------
META_ROWS = 5
h8_data = h8.iloc[META_ROWS:].copy()
h8_data = h8_data.rename(columns={"Series Description": "period"})

for col in h8_data.columns:
    if col != "period":
        h8_data[col] = pd.to_numeric(h8_data[col], errors="coerce")

h8_data["quarter"] = pd.PeriodIndex(h8_data["period"], freq="Q")

# ----------------------------- CLEAN FEDFUNDS -----------------------------
fed["observation_date"] = pd.to_datetime(fed["observation_date"])
fed["quarter"] = fed["observation_date"].dt.to_period("Q")
fed_q = fed.groupby("quarter", as_index=False)["FEDFUNDS"].mean()

# ----------------------------- MERGE -----------------------------
merged = pd.merge(h8_data, fed_q, on="quarter", how="inner")
merged = merged.sort_values("quarter").reset_index(drop=True)

numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c != "FEDFUNDS"]

X = merged[feature_cols].values
y = merged["FEDFUNDS"].values

# ----------------------------- TRAIN / TEST -----------------------------
split = int(len(X) * 0.9)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

quarters_test = merged["quarter"].iloc[split:].astype(str).values

# ----------------------------- MODEL -----------------------------
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestRegressor(n_estimators=300, random_state=42,
                                 min_samples_leaf=2, n_jobs=-1))
])

model.fit(X_train, y_train)
raw_pred = model.predict(X_test)

# ----------------------------- NEW: LIMIT CHANGE -----------------------------
limited_pred = raw_pred.copy()

for i in range(1, len(limited_pred)):
    prev = limited_pred[i-1]
    diff = limited_pred[i] - prev

    if diff > 0.75:
        limited_pred[i] = prev + 0.75
    elif diff < -0.75:
        limited_pred[i] = prev - 0.75

# ----------------------------- METRICS -----------------------------
mse = mean_squared_error(y_test, limited_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, limited_pred)
r2 = r2_score(y_test, limited_pred)

print("\nModel with Change Constraint:")
print(f"RMSE: {rmse:.3f}")
print(f"MAE:  {mae:.3f}")
print(f"R^2:  {r2:.3f}")

# ----------------------------- RESULTS DF -----------------------------
results = pd.DataFrame({
    "quarter": quarters_test,
    "actual": y_test,
    "predicted_raw": raw_pred,
    "predicted_limited": limited_pred,
})

results["error"] = results["actual"] - results["predicted_limited"]

# ----------------------------- PLOT 1: ACTUAL vs PRED -----------------------------
plt.figure(figsize=(10,5))
plt.plot(results["quarter"], results["actual"], label="Actual", marker="o")
plt.plot(results["quarter"], results["predicted_limited"], label="Predicted (Limited)", marker="x")
plt.xticks(rotation=45, ha="right")
plt.title("Actual vs Limited Predicted Federal Funds Rate")
plt.xlabel("Quarter")
plt.ylabel("Fed Funds Rate")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------- PLOT 2: ERROR -----------------------------
plt.figure(figsize=(10,5))
plt.plot(results["quarter"], results["error"], marker="o")
plt.axhline(0, linestyle="--")
plt.xticks(rotation=45, ha="right")
plt.title("Prediction Error (Actual − Predicted)")
plt.xlabel("Quarter")
plt.ylabel("Error")
plt.tight_layout()
plt.show()
