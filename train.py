import os
import json
import joblib
import pandas as pd
import numpy as np

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
# EXP-07: Random Forest with selected features
# -----------------------

# Correlation-based feature selection
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X_selected = X.drop(columns=to_drop)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
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
joblib.dump(pipeline, "output/model_EXP-07.pkl")

# Save metrics
results = {
    "mse": mse,
    "r2_score": r2
}

with open("output/results_EXP-07.json", "w") as f:
    json.dump(results, f, indent=4)
