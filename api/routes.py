import os
import shutil
import uuid
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from config import logger, UPLOAD_DIR
from models.data_models import PredictionResult, TaskStatus, GPUStatus
from services.scan_service import process_scan_task, get_task_result, PROCESSING_TASKS
from api.dependencies import get_gpu_status

router = APIRouter(prefix="/api")

@router.post("/analyze", response_model=TaskStatus)
async def analyze_scan(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Endpoint to upload and analyze a CTPA scan"""
    try:
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize task status
        PROCESSING_TASKS[task_id] = {"status": "queued"}
        
        # Add task to background processing
        background_tasks.add_task(process_scan_task, task_id, file_path)
        
        logger.info(f"Task {task_id} created for file {file.filename}")
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Scan upload successful, processing started"
        }
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}", response_model=PredictionResult)
async def get_task_status(task_id: str):
    """Get status of a processing task"""
    result = get_task_result(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return result

@router.get("/gpu-status", response_model=GPUStatus)
async def gpu_status():
    """Get GPU status information"""
    return get_gpu_status()