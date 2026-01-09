import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Create output folder
os.makedirs("output", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Experiment name
experiment_name = "Model-RandomForest, Data Split Strategy"

print(f"=== {experiment_name} ===")

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics (for GitHub Actions logs)
print("MSE:", mse)
print("R2 Score:", r2)

# Save model
joblib.dump(model, "output/model.pkl")

# Save metrics
results = {
    "experiment": experiment_name,
    "mse": mse,
    "r2_score": r2
}

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)

# GitHub Actions Job Summary
summary = f"""
## Experiment Results

**Experiment:** {experiment_name}  
**MSE:** {mse:.4f}  
**R² Score:** {r2:.4f}  
"""

# Write to GitHub Actions summary
summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
if summary_file:
    with open(summary_file, "a") as f:
        f.write(summary)

print("Experiment completed successfully.")
