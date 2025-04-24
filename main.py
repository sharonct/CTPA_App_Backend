from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.health import router as health_router
from api.analysis import router as analysis_router
from api.upload import router as upload_router
from api.scans import router as scans_router
from api.report import router as report_router
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create the FastAPI app
app = FastAPI(
    title="CTPA Analysis API",
    description="API for analyzing CT Pulmonary Angiography scans",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your Streamlit app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logging.info(f"Response status: {response.status_code}")
    return response

# Include routers
app.include_router(health_router, tags=["health"])
app.include_router(upload_router, tags=["upload"])
app.include_router(scans_router, tags=["scans"])
app.include_router(analysis_router, tags=["analysis"])
app.include_router(report_router, tags=["report"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CTPA Analysis API",
        "docs_url": "/docs",
        "version": "1.0.0"
    }

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)