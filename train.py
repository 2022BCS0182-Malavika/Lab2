import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

# Experiment 7: Random Forest Data Split Strategy
# Experiment: Lasso Regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])


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
joblib.dump(pipeline, "output/model.pkl")

# Save metrics
results = {
    "mse": mse,
    "r2_score": r2
}

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)
