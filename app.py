from fastapi import FastAPI, Request
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from uvicorn import run as app_run
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os

load_dotenv()

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.constant.application import *

import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

templates = Jinja2Templates(directory='templates')

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# FORM CLASS
# =========================
class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.text: Optional[str] = None

    async def get_text_data(self):
        form = await self.request.form()
        self.text = form.get('input_text')


# =========================
# TRAIN ROUTE
# =========================
@app.get("/train")
async def train_model():
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        return Response("✅ Training successful!")
    except Exception as e:
        return Response(f"❌ Error: {e}")


# =========================
# HOME PAGE
# =========================
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# =========================
# PREDICT PAGE (GET)
# =========================
@app.get("/predict")
async def predict_page(request: Request):
    return templates.TemplateResponse(
        "prediction.html",
        {"request": request, "context": False}
    )


# =========================
# PREDICTION (POST)
# =========================
@app.post("/predict")
async def predict(request: Request):
    try:
        form = DataForm(request)
        await form.get_text_data()

        input_data = [form.text]

        model_path = os.path.join("artifacts", "model.pkl")

        if not os.path.exists(model_path):
            return templates.TemplateResponse(
                "prediction.html",
                {
                    "request": request,
                    "context": True,
                    "prediction": "⚠️ Train model first!"
                }
            )

        prediction_pipeline = PredictionPipeline()
        prediction = prediction_pipeline.run_pipeline(input_data=input_data)

        result = int(prediction[0])

        return templates.TemplateResponse(
            "prediction.html",
            {
                "request": request,
                "context": True,
                "prediction": result
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "prediction.html",
            {
                "request": request,
                "context": True,
                "prediction": f"❌ Error: {str(e)}"
            }
        )


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)