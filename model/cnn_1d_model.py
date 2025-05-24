import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Load data
df = pd.read_csv("data/sensor_data.csv")
X = df[["radius", "refractive_index"]].values
y = df["absorption_peak_freq"].values

# Reshape for 1D CNN: (samples, time_steps, channels)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define 1D CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X.shape[1], 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer for regression
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train model
history = model.fit(X_train, y_train, epochs=100, verbose=1, validation_split=0.1)

# Predict on test set
y_pred = model.predict(X_test).flatten()

# Evaluate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Output folder
os.makedirs("output", exist_ok=True)

# Save evaluation metrics
with open("output/cnn_results.txt", "w") as f:
    f.write("1D CNN Evaluation Results\n")
    f.write(f"R² Score: {r2:.4f}\n")
    f.write(f"MSE: {mse:.6f}\n")
    f.write(f"MAE: {mae:.6f}\n")

# Scatter plot of predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='teal', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal Fit')
plt.xlabel("Actual Frequency")
plt.ylabel("Predicted Frequency")
plt.title("1D CNN - Predicted vs Actual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/cnn_prediction_plot.png")

# Training & validation loss plot
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.title("1D CNN - Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/cnn_training_curve.png")