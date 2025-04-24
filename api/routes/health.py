from fastapi import APIRouter
from config import DEVICE
from models.ct_clip_loader import model_loaded

router = APIRouter()

@router.get("/health")
async def health_check():
    """Check if the API is running and models are loaded"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded(),
        "device": str(DEVICE),
        "api_url": "http://localhost:8000"
    }