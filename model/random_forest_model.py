from sklearn.ensemble import RandomForestRegressor


# === Random Forest Regressor ===
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)
rf_pred = rf_model.predict(X)

rf_r2 = r2_score(y, rf_pred)
print(f"Random Forest R² Score: {rf_r2:.4f}")

# Save performance
with open("output/model_performance.txt", "a") as f:
    f.write(f"\nRandom Forest Regressor\nR² Score: {rf_r2:.4f}\n")

# Plot predictions
plt.figure(figsize=(8, 6))
plt.scatter(y, rf_pred, color='green', label='Random Forest')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Ideal')
plt.xlabel("Actual Absorption Frequency (THz)")
plt.ylabel("Predicted Absorption Frequency (THz)")
plt.title("Random Forest - Absorption Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/rf_prediction_plot.png")
print("Random Forest plot saved to output/rf_prediction_plot.png")

