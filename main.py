from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import os

app = FastAPI(title="Logistic Regression Deployment")

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "logistic_model.pkl")
model = joblib.load(MODEL_PATH)

templates = Jinja2Templates(directory="templates")


# ---------- JSON API endpoint ----------
@app.get("/")
def root():
    return {"message": "Logistic Regression API is running. Go to /form for the web UI or POST to /predict."}


@app.post("/predict")
def predict(hours_studied: float):
    prediction = model.predict(np.array([[hours_studied]]))
    probability = model.predict_proba(np.array([[hours_studied]]))

    result = "Pass" if prediction[0] == 1 else "Fail"

    return {
        "hours_studied": hours_studied,
        "prediction": int(prediction[0]),
        "result": result,
        "probability_fail": round(float(probability[0][0]), 4),
        "probability_pass": round(float(probability[0][1]), 4),
    }


# ---------- HTML form (optional UI) ----------
@app.get("/form", response_class=HTMLResponse)
def show_form(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": None}
    )


@app.post("/form", response_class=HTMLResponse)
def submit_form(request: Request, hours_studied: float = Form(...)):
    prediction = model.predict(np.array([[hours_studied]]))
    probability = model.predict_proba(np.array([[hours_studied]]))

    result = "Pass ✅" if prediction[0] == 1 else "Fail ❌"
    prob_pass = round(float(probability[0][1]) * 100, 2)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "hours_studied": hours_studied,
            "prob_pass": prob_pass,
        },
    )