import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

df = pd.read_csv("data/sensor_data.csv")
X = df[["radius", "refractive_index"]]
y = df["absorption_peak_freq"]

model = KNeighborsRegressor(n_neighbors=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

os.makedirs("output", exist_ok=True)
with open("output/KNN_Regression/knn_results.txt", "w") as f:
    f.write(f"KNN Regression\nR²: {r2:.4f}\nMSE: {mse:.6f}\nMAE: {mae:.6f}\n")

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='green', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal Fit')
plt.xlabel("Actual Frequency")
plt.ylabel("Predicted Frequency")
plt.title("KNN Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/KNN_Regression/knn_plot.png")