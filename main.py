# FastAPI app (loads artifacts, serves predictions)
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import pickle
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model and label encoder
model = pickle.load(open("model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "prediction": None})

@app.post("/", response_class=HTMLResponse)
async def form_post(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):

    # Convert to DataFrame with column names
    df = pd.DataFrame([[
        sepal_length, sepal_width, petal_length, petal_width
    ]], columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])

    prediction_idx = model.predict(df)[0]
    prediction = label_encoder.inverse_transform([prediction_idx])[0]

    return templates.TemplateResponse("form.html", {
        "request": request,
        "prediction": f"Predicted class: {prediction}"
    })
