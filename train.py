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

# -----------------------
# EXP-06: Random Forest – 100 trees, max depth=15
# -----------------------

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
pipeline = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print("MSE:", mse)
print("R2 Score:", r2)

# Save model
joblib.dump(pipeline, "output/model_EXP-06.pkl")

# Save metrics
results = {
    "mse": mse,
    "r2_score": r2
}

with open("output/results_EXP-06.json", "w") as f:
    json.dump(results, f, indent=4)
