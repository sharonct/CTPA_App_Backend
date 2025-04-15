from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from api.routes import health, scans, analysis
from api.routes.report import router as report_router
from utils.logging import logger
from models.ct_clip_loader import load_ctclip_model
from models.vqa_model import load_vqa_model
from models.ct_report_model import load_ct_report_model
from config import ENABLE_CORS, DEBUG_MODE, USE_VQA_MODEL, USE_CT_REPORT_MODEL

# Initialize FastAPI app
app = FastAPI(
    title="CTPA Report Generator API",
    description="API for CT Pulmonary Angiography Report Generation",
    version="1.0.0",
    debug=DEBUG_MODE
)

# Add CORS middleware if enabled
if ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include the API routes
app.include_router(health.router, tags=["Health"])
app.include_router(scans.router, tags=["Scans"])
app.include_router(analysis.router, tags=["Analysis"])
app.include_router(report_router, prefix="/report", tags=["Reports"])

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting up the application")
    
    # Load CT-CLIP model
    load_ctclip_model()
    
    # Load VQA model if enabled
    if USE_VQA_MODEL:
        load_vqa_model()
    
    # Load CT Report model if enabled
    if USE_CT_REPORT_MODEL:
        load_ct_report_model()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)