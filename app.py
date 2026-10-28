from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Mammal Sleep Prediction API")

artifact = joblib.load(Path(__file__).resolve().with_name("model.joblib"))
# Support both dict-wrapped artifact and direct pipeline object
model = artifact["model"] if isinstance(artifact, dict) else artifact
model_params = artifact.get("parameters", {}) if isinstance(artifact, dict) else model.get_params()

class MammalInput(BaseModel):
    body_weight: float
    brain_weight: float
    max_life_span: float
    gestation_time: float
    predation_index: int
    sleep_exposure_index: int
    danger_index: int

@app.get("/")
def health_check():
    return {"status": "ok", "message": "FastAPI Model Server is running"}

@app.get("/parameters")
def get_model_parameters():
    return {"model_parameters": str(model_params)}

@app.post("/predict")
def predict(data: MammalInput):
    df_input = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df_input)[0]
    return {
        "input": data.model_dump(),
        "predicted_total_sleep_hours": round(float(prediction), 2)
    }