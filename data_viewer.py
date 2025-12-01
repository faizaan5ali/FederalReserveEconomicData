import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read CSV and clean
csv_path = "FRB_H8.csv"
df = pd.read_csv(csv_path, header=4)
df.rename(columns={df.columns[0]: "Time Period"}, inplace=True)

df["Time Period"] = df["Time Period"].astype(str).str.strip()
df = df[df["Time Period"].str.match(r"^\d{4}Q[1-4]$")]
df["Time Period"] = pd.PeriodIndex(df["Time Period"], freq="Q").to_timestamp()
df.set_index("Time Period", inplace=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nSUMMARY STATISTICS")
print(df.describe())
print(len(df))

# plot histograms
for col in df.columns:
    plt.figure(figsize=(8, 4))
    plt.hist(df[col].dropna(), bins=30)
    plt.title(f"Histogram of {col}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

# boxplots
for col in df.columns:
    plt.figure(figsize=(6, 4))
    plt.boxplot(df[col].dropna(), vert=True)
    plt.title(f"Boxplot of {col}")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
    break

# correlation heatmap
plt.figure(figsize=(14, 10))
corr = df.corr()

# Use imshow instead of seaborn
plt.imshow(corr, cmap="viridis", aspect="auto")
plt.colorbar(label="Correlation")

plt.xticks(ticks=np.arange(len(corr.columns)), labels=corr.columns, rotation=90)
plt.yticks(ticks=np.arange(len(corr.columns)), labels=corr.columns)

plt.title("Correlation Heatmap of Bank Credit Variables")
plt.tight_layout()
plt.show()

# time series for Bank Credit feature
target_col = "H8/H8/B1001NCBCQG"
title = "Bank credit"

if target_col not in df.columns:
    print(f"\nColumn '{target_col}' not found in dataset. Available columns:")
    print(df.columns.tolist())
else:
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df[target_col], linewidth=2)
    plt.title(f"{title} Over Time")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
