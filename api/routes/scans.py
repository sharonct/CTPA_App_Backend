import os
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Query
from typing import List
from api.schemas import ScanMetadata, SliceResponse
from services.scan_service import upload_scan, get_slice
from utils.file_utils import load_metadata
from utils.logging import logger
from config import UPLOAD_DIR

router = APIRouter()

@router.post("/upload", response_model=ScanMetadata)
async def upload_scan_endpoint(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload a CT scan file (NIFTI format)
    """
    try:
        return await upload_scan(file, background_tasks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@router.get("/scans", response_model=List[ScanMetadata])
async def list_scans():
    """List all uploaded scans"""
    scans = []
    
    try:
        for scan_id in os.listdir(UPLOAD_DIR):
            metadata = load_metadata(scan_id)
            
            if metadata:
                scans.append(ScanMetadata(
                    scan_id=metadata["scan_id"],
                    filename=metadata["filename"],
                    upload_time=metadata["upload_time"],
                    status=metadata["status"],
                    dimensions=metadata.get("dimensions"),
                    pe_probability=metadata.get("pe_probability")
                ))
    
    except Exception as e:
        logger.error(f"Error listing scans: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing scans: {str(e)}")
    
    return scans

@router.get("/scans/{scan_id}", response_model=ScanMetadata)
async def get_scan(scan_id: str):
    """Get metadata for a specific scan"""
    metadata = load_metadata(scan_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Scan with ID {scan_id} not found")
    
    try:
        return ScanMetadata(
            scan_id=metadata["scan_id"],
            filename=metadata["filename"],
            upload_time=metadata["upload_time"],
            status=metadata["status"],
            dimensions=metadata.get("dimensions"),
            pe_probability=metadata.get("pe_probability")
        )
    
    except Exception as e:
        logger.error(f"Error getting scan {scan_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting scan: {str(e)}")

@router.get("/slice/{scan_id}", response_model=SliceResponse)
async def get_slice_endpoint(
    scan_id: str, 
    view: str = Query("axial", description="View type: axial, sagittal, or coronal"), 
    slice_idx: int = Query(0, description="Slice index"),
    window_center: int = Query(-600, description="Window center (HU)"),
    window_width: int = Query(1500, description="Window width (HU)")
):
    """Get a specific slice from a scan"""
    try:
        return get_slice(scan_id, view, slice_idx, window_center, window_width)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting slice for scan {scan_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error getting slice: {str(e)}")