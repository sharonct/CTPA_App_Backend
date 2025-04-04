import os
import torch
import nibabel as nib
from datetime import datetime
import time
from config import logger
from services.model_service import get_model
from utils.preprocessing import preprocess_scan

# Storage for processing tasks
PROCESSING_TASKS = {}

async def process_scan_task(task_id: str, file_path: str):
    """Process a scan file in the background"""
    try:
        PROCESSING_TASKS[task_id]["status"] = "processing"
        start_time = time.time()
        
        # Load the NIfTI file
        img = nib.load(file_path)
        scan_data = img.get_fdata()
        
        # Get the model
        model = get_model()
        
        # Preprocess the scan
        processed_data = preprocess_scan(scan_data)
        
        # Run inference
        with torch.no_grad():
            predictions = model(processed_data)
        
        # Process predictions
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        # Example: Extract PE probability (adjust based on your model)
        pe_probability = float(predictions[0][0])
        pe_present = pe_probability > 0.5
        
        # Sample location data for demonstration - replace with actual logic
        pe_locations = []
        if pe_present:
            pe_locations = [
                {
                    "artery": "Right lower lobe pulmonary artery",
                    "confidence": 0.92,
                    "severity": "moderate"
                }
            ]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update task status
        PROCESSING_TASKS[task_id] = {
            "status": "completed",
            "result": {
                "task_id": task_id,
                "status": "completed",
                "pe_present": pe_present,
                "pe_probability": pe_probability,
                "pe_locations": pe_locations,
                "processing_time": processing_time,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        logger.info(f"Task {task_id} completed in {processing_time:.2f} seconds")
        
        # Clean up the file
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to remove temp file {file_path}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        PROCESSING_TASKS[task_id] = {
            "status": "failed",
            "result": {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }

def get_task_result(task_id: str):
    """Get the current status or result of a task"""
    if task_id not in PROCESSING_TASKS:
        return None
    
    task = PROCESSING_TASKS[task_id]
    status = task["status"]
    
    if status == "completed" or status == "failed":
        return task["result"]
    else:
        return {
            "task_id": task_id,
            "status": status
        }