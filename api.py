from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import joblib

app = FastAPI(title="Repair vs Replacement AI API", version="1.0")
cost_model = joblib.load("repair_cost_model.joblib")
decision_model = joblib.load("repair_replace_model.joblib")

class DeviceInput(BaseModel):
    Device_Type: str
    Brand: str
    Device_Age: float = Field(ge=0)
    Problem_Category: str
    Physical_Condition: str
    Previous_Repairs: str
    Repair_Count: int = Field(ge=0)
    Damaged_Components: str
    Purchase_Price: float = Field(gt=0)
    Warranty_Status: str = "Unknown"

@app.get("/")
def home():
    return {"message": "Repair vs Replacement AI API is running"}

@app.post("/predict")
def predict(item: DeviceInput):
    row = item.model_dump()
    cost_input = pd.DataFrame([row])
    estimated_cost = float(cost_model.predict(cost_input)[0])
    row["Predicted_Repair_Cost"] = estimated_cost
    decision_input = pd.DataFrame([row])
    decision = str(decision_model.predict(decision_input)[0])
    probabilities = decision_model.predict_proba(decision_input)[0]
    classes = list(decision_model.named_steps["model"].classes_)
    confidence = float(max(probabilities))
    return {
        "predicted_repair_cost": round(max(0, estimated_cost), 2),
        "repair_replacement_result": decision,
        "prediction_confidence": round(confidence, 4),
        "warning": "Prototype trained on 20 synthetic sample records; not suitable for real customer decisions."
    }
