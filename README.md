# Repair vs Replacement AI – trained prototype

## Models
- `repair_cost_model.joblib`: regression model; estimates repair cost in INR.
- `repair_replace_model.joblib`: classification model; predicts Repair or Replace.
- `api.py`: FastAPI endpoint that runs regression first, then classification.

## Training data
The trainer joined the four Zoho CSV exports using `Device_Name` / `Device_ID`.
Regression target: `Actual_Repair_Cost` from Repair History.
Classification target: `Repair_Replacement_Result` from the sample AI Prediction records.

## Run on Windows (VS Code terminal)
1. Extract this ZIP.
2. Open the extracted folder in VS Code.
3. Run:
   `py -m venv venv`
   `venv\Scripts\activate`
   `pip install -r requirements.txt`
   `uvicorn api:app --reload`
4. Open `http://127.0.0.1:8000/docs`.
5. Test `POST /predict` with an example:
```json
{
  "Device_Type": "Laptop",
  "Brand": "Dell",
  "Device_Age": 3,
  "Problem_Category": "Display",
  "Physical_Condition": "Poor",
  "Previous_Repairs": "No",
  "Repair_Count": 0,
  "Damaged_Components": "Screen",
  "Purchase_Price": 58000,
  "Warranty_Status": "Expired"
}
```

## Important limitation
Only 20 synthetic demonstration records were used. Cross-validation numbers in
`training_report.json` are unstable and are not evidence of real-world accuracy.
Collect hundreds or thousands of genuine labelled repair cases and evaluate on
a separate held-out test set before using predictions for customer decisions.
