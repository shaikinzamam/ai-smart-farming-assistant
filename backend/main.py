import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.chatbot import FarmerChatbot
from backend.model import DummyPlantDiseaseModel
from backend.utils import get_solution_bundle, load_solutions

import uvicorn


BASE_DIR = Path(__file__).resolve().parent.parent
SOLUTIONS_PATH = BASE_DIR / "data" / "solutions.json"
MODEL_PATH = BASE_DIR / "models" / "plant_model.h5"


class ChatRequest(BaseModel):
    message: str


app = FastAPI(
    title="AI Smart Farming Assistant API",
    description="Hackathon prototype for plant disease detection and farmer support.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

solutions = load_solutions(SOLUTIONS_PATH)
plant_model = DummyPlantDiseaseModel(model_path=MODEL_PATH, solutions=solutions)
chatbot = FarmerChatbot()


@app.get("/")
async def root() -> dict:
    return {
        "message": "AI Smart Farming Assistant backend is running.",
        "predict_endpoint": "/predict",
        "chat_endpoint": "/chat",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    disease = plant_model.predict(image_bytes=image_bytes, filename=file.filename or "uploaded-image")
    solution_bundle = get_solution_bundle(solutions=solutions, disease=disease)

    return {
        "filename": file.filename,
        "disease": disease,
        "organic_solution": solution_bundle["organic_solution"],
        "chemical_solution": solution_bundle["chemical_solution"],
        "confidence": solution_bundle["confidence_hint"],
    }


@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    response = chatbot.get_response(message)
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
