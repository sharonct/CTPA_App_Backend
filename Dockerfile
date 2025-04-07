from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from utils.logging import logger
from models.ct_clip_loader import load_ctclip_model

# Initialize FastAPI app
app = FastAPI(
    title="CT-CLIP VQA API",
    description="API for CT Scan VQA with CT-CLIP and BiomedVLP-CXR-BERT",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API routes
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting up the application")
    
    # Start model loading in the background
    load_ctclip_model()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)