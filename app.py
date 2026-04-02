from fastapi import FastAPI, Request
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from uvicorn import run as app_run
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os

load_dotenv()

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.constant.application import APP_HOST, APP_PORT

import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

templates = Jinja2Templates(directory='templates')
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ ===== DYNAMIC MODEL LOADING =====
ARTIFACTS_DIR = os.path.join("src", "artifacts")
def get_latest_paths():
    folders = [
        f for f in os.listdir(ARTIFACTS_DIR)
        if os.path.isdir(os.path.join(ARTIFACTS_DIR, f))
    ]

    if not folders:
        raise Exception("No training folders found!")

    latest_folder = sorted(folders)[-1]
    base_path = os.path.join(ARTIFACTS_DIR, latest_folder)

    # ✅ CORRECT PATHS (BASED ON YOUR STRUCTURE)
    model_path = os.path.join(
        base_path, "model_evaluation", "best_model.pkl"
    )

    vectorizer_path = os.path.join(
        base_path, "data_transformation", "transformed_object", "vectorizer.pkl"
    )

    # 🔍 DEBUG
    print("\n===== FINAL PATH DEBUG =====")
    print("Model path:", model_path)
    print("Vectorizer path:", vectorizer_path)
    print("Model exists:", os.path.exists(model_path))
    print("Vectorizer exists:", os.path.exists(vectorizer_path))
    print("===========================\n")

    return model_path, vectorizer_path, base_path

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.text: Optional[str] = None

    async def get_text_data(self):
        form = await self.request.form()
        self.text = form.get('input_text')


# ── Home ─────────────────────────
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ── Train ────────────────────────
@app.get("/train")
async def train_model(request: Request):
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "train_msg": "Training completed successfully!"}
        )
    except Exception as e:
        return Response(f"Training error: {e}", status_code=500)


# ── Predict Page ─────────────────
@app.get("/predict")
async def predict_page(request: Request):
    return templates.TemplateResponse(
        "prediction.html",
        {"request": request, "context": False, "scanned_text": ""}
    )


# ── Predict API ──────────────────
@app.post("/predict")
async def predict(request: Request):
    try:
        form = DataForm(request)
        await form.get_text_data()

        input_text = form.text or ""

        # ✅ Get latest model paths
        model_path, vectorizer_path, base_path = get_latest_paths()

        # ❌ If files missing
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            return templates.TemplateResponse(
                "prediction.html",
                {
                    "request": request,
                    "context": True,
                    "prediction": "⚠️ Model or Vectorizer not found. Train again.",
                    "scanned_text": input_text,
                }
            )

        pipeline = PredictionPipeline(
            model_path=model_path,
            vectorizer_path=vectorizer_path
        )

        prediction = pipeline.run_pipeline(input_text)

        # 🔍 DEBUG OUTPUT
        print("RAW PREDICTION:", prediction)

        result = int(prediction[0])

        print("FINAL RESULT:", result)

        return templates.TemplateResponse(
            "prediction.html",
            {
                "request": request,
                "context": True,
                "prediction": result,
                "scanned_text": input_text,
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "prediction.html",
            {
                "request": request,
                "context": True,
                "prediction": f"❌ Error: {str(e)}",
                "scanned_text": "",
            }
        )


@app.get("/inbox", response_class=HTMLResponse)
async def inbox_page(request: Request):
    return templates.TemplateResponse("inbox.html", {"request": request})


@app.get("/spam", response_class=HTMLResponse)
async def spam_page(request: Request):
    return templates.TemplateResponse("spam.html", {"request": request})

# ── Run ─────────────────────────
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)