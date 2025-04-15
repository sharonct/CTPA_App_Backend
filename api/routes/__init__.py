from fastapi import APIRouter
from api.routes.health import router as health_router
from api.routes.scans import router as scans_router
# from api.routes.analysis import router as analysis_router

# Create the main router
router = APIRouter()

# Include the different route modules
router.include_router(health_router, tags=["Health"])
router.include_router(scans_router, tags=["Scans"])
# router.include_router(analysis_router, tags=["Analysis"])