import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Allow running as  python backend/main.py  from the project root.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.chatbot import FarmerChatbot
from backend.model import DISPLAY_NAMES, PlantDiseaseModel
from backend.utils import get_solution_bundle, load_solutions

import uvicorn

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
SOLUTIONS_PATH = BASE_DIR / "data" / "solutions.json"
MODEL_PATH     = BASE_DIR / "models" / "plant_model.h5"
METADATA_PATH  = BASE_DIR / "models" / "model_metadata.json"

# ─── Schemas ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class PredictResponse(BaseModel):
    filename:         str
    disease:          str
    organic_solution: str
    chemical_solution: str
    confidence:       str
    class_key:        str
    model_used:       str

class ChatResponse(BaseModel):
    response: str

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Smart Farming Assistant API",
    description=(
        "Plant disease detection (Tomato & Banana) and NLP farming assistant. "
        "Powered by MobileNetV2 transfer learning."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup: load model and data ─────────────────────────────────────────────

solutions    = load_solutions(SOLUTIONS_PATH)
plant_model  = PlantDiseaseModel(model_path=MODEL_PATH, metadata_path=METADATA_PATH)
chatbot      = FarmerChatbot()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> dict:
    return {
        "service": "AI Smart Farming Assistant",
        "version": "2.0.0",
        "status":  "running",
        "endpoints": {
            "predict":    "/predict   [POST] — upload a plant image",
            "chat":       "/chat      [POST] — ask a farming question",
            "classes":    "/classes   [GET]  — list disease classes",
            "model_info": "/model-info [GET] — model metadata",
        },
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> dict:
    """
    Accepts a JPG / PNG / WEBP plant image and returns:
    - Predicted disease name
    - Organic and chemical treatment recommendations
    - Model confidence
    """
    # ── Validate file type ────────────────────────────────────────────────────
    ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
    content_type  = (file.content_type or "").lower()

    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Please upload JPG, PNG, or WEBP.",
        )

    # ── Read image ────────────────────────────────────────────────────────────
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > 10 * 1024 * 1024:  # 10 MB guard
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit.")

    # ── Run inference ─────────────────────────────────────────────────────────
    result    = plant_model.predict(image_bytes=image_bytes, filename=file.filename or "image")
    disease   = result["disease"]

    # ── Look up solutions ─────────────────────────────────────────────────────
    # solutions.json uses display names as keys
    bundle = get_solution_bundle(solutions=solutions, disease=disease)

    return {
        "filename":          file.filename or "uploaded-image",
        "disease":           disease,
        "organic_solution":  bundle["organic_solution"],
        "chemical_solution": bundle["chemical_solution"],
        "confidence":        result["confidence"],
        "class_key":         result.get("class_key", ""),
        "model_used":        result.get("model_used", "unknown"),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> dict:
    """Accepts a plain-text farming question and returns a helpful response."""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(message) > 1000:
        raise HTTPException(status_code=400, detail="Message exceeds 1000-character limit.")

    response = chatbot.get_response(message)
    return {"response": response}


@app.get("/classes")
async def list_classes() -> dict:
    """Returns all disease classes the model can predict."""
    return {
        "classes": [
            {"key": key, "display": display}
            for key, display in DISPLAY_NAMES.items()
        ]
    }


@app.get("/model-info")
async def model_info() -> dict:
    """Returns metadata about the loaded model."""
    import json
    meta_path = METADATA_PATH
    if meta_path.exists():
        with meta_path.open() as f:
            meta = json.load(f)
        return {
            "model_loaded":       MODEL_PATH.exists(),
            "num_classes":        meta.get("num_classes"),
            "best_val_accuracy":  meta.get("best_val_accuracy"),
            "image_size":         meta.get("image_size"),
        }
    return {
        "model_loaded": MODEL_PATH.exists(),
        "note":         "Model metadata not found. Train the model first: python train.py",
    }


# ─── Dev server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )