import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("data/sensor_data.csv")
X = df[["radius", "refractive_index"]]
y = df["absorption_peak_freq"]

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train model
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Output folder
os.makedirs("output", exist_ok=True)

# Save performance
with open("output/Polynomial_Regression/polynomial_results.txt", "w") as f:
    f.write(f"Polynomial Regression\nR²: {r2:.4f}\nMSE: {mse:.6f}\n")

# Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='green', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal Fit')
plt.xlabel("Actual Frequency")
plt.ylabel("Predicted Frequency")
plt.title("Polynomial Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/Polynomial_Regression/polynomial_regression_plot.png")