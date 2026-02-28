from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("output/model.pkl")

class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def root():
    return {"message": "Wine Quality API is running"}

@app.post("/predict")
def predict(data: WineInput):
    input_data = np.array([[ 
        data.fixed_acidity,
        data.volatile_acidity,
        data.citric_acid,
        data.residual_sugar,
        data.chlorides,
        data.free_sulfur_dioxide,
        data.total_sulfur_dioxide,
        data.density,
        data.pH,
        data.sulphates,
        data.alcohol
    ]])

    prediction = model.predict(input_data)[0]

    return {
        "name": "Badam Malavika",
        "roll_no": "2022BCS0182",
        "wine_quality": int(prediction)
    }
