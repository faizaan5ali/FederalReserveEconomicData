import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

# load data and clean
csv_path = "FRB_H8.csv"
df = pd.read_csv(csv_path, header=4)
df.rename(columns={df.columns[0]: "Time Period"}, inplace=True)

df["Time Period"] = df["Time Period"].astype(str).str.strip()
df = df[df["Time Period"].str.match(r"^\d{4}Q[1-4]$")]
df["Time Period"] = pd.PeriodIndex(df["Time Period"], freq="Q").to_timestamp()
df.set_index("Time Period", inplace=True)

# convert numeric columns
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# target var
target_col = "H8/H8/B1001NCBCQG"
ts = df[target_col].dropna()

# training model
train = ts[:-8]
test = ts[-8:]

auto_model = pm.auto_arima(train, seasonal=False, stepwise=True,
                           suppress_warnings=True, error_action='ignore',
                           max_p=5, max_q=5, d=None, trace=True)

print("\nAUTO ARIMA SELECTED PARAMETERS")
print(auto_model.summary())

best_order = auto_model.order
model = ARIMA(train, order=best_order)
model_fit = model.fit()
print("\nARIMA MODEL FIT SUMMARY")
print(model_fit.summary())

# forecast into test subset
forecast = model_fit.forecast(steps=len(test))

# metrics
rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
mape = np.mean(np.abs((test - forecast) / test)) * 100
r2 = r2_score(test, forecast)

print("\nTEST SET METRICS")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.4f}")

# plot actual trend vs predicted
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, forecast, label="Forecast")
plt.title("B1001NCBCQG - ARIMA Forecast (Test Period)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# residuals
residuals = test - forecast
plt.figure(figsize=(10, 4))
plt.plot(test.index, residuals, marker='o', linestyle='-', color='red')
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Residuals (Actual - Forecast)")
plt.xlabel("Time")
plt.ylabel("Residual")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# test out the model some more by predicting past the dataset points
future_steps = 8
future_forecast = model_fit.forecast(steps=future_steps)
future_index = pd.date_range(start=ts.index[-1], periods=future_steps + 1, freq="Q")[1:]

plt.figure(figsize=(12, 6))
plt.plot(ts.index, ts, label="Historical")
plt.plot(future_index, future_forecast, label="Future Forecast", linestyle="--")
plt.title("B1001NCBCQG - Future Forecast (Next 2 Years)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

print("\nFUTURE FORECAST (NEXT 8 QUARTERS)")
for date, value in zip(future_index, future_forecast):
    print(f"{date.date()}: {value:.4f}")
