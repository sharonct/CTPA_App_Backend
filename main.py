import os
import uvicorn
import shutil
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import logger, UPLOAD_DIR, MODEL_PATH, PORT
from api.routes import router
from services.model_service import clear_model_cache

app = FastAPI(title="CTPA Analysis Backend API")

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to only allow your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Run when the server starts up"""
    logger.info("Backend server starting up")
    
    # Ensure the upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Check for GPU
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available, using CPU")
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
    else:
        logger.info(f"Model file found: {MODEL_PATH}")

@app.on_event("shutdown")
async def shutdown_event():
    """Run when the server shuts down"""
    logger.info("Backend server shutting down")
    
    # Clear model cache
    clear_model_cache()
    
    # Clean up temporary files
    try:
        shutil.rmtree(UPLOAD_DIR)
    except Exception as e:
        logger.error(f"Error cleaning up upload directory: {str(e)}")

if __name__ == "__main__":
    # Run the server
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)