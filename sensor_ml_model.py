import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import os

# Set up directories
os.makedirs("output", exist_ok=True)

# Load data
data = pd.read_csv("data/mock_sensor_data.csv")
X = data[["radius", "refractive_index"]]
y = data["absorption_peak_freq"]

# Polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

# Evaluate
r2 = r2_score(y, y_pred)

# Save performance to file
with open("output/model_performance.txt", "w") as f:
    f.write(f"Polynomial Regression Model\n")
    f.write(f"R² Score: {r2:.4f}\n")

print(f"Model trained. R² Score: {r2:.4f}")

# Plot predictions vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Ideal Line')
plt.xlabel("Actual Absorption Frequency (THz)")
plt.ylabel("Predicted Absorption Frequency (THz)")
plt.title("Polynomial Regression - Absorption Peak Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/prediction_plot.png")
print("Plot saved to output/prediction_plot.png")
