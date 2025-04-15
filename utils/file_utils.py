import os
import json
import nibabel as nib
import numpy as np
import shutil
from datetime import datetime
from utils.logging import logger
from config import UPLOAD_DIR

def save_uploaded_file(scan_id, file):
    """Save an uploaded file to the scan directory"""
    scan_dir = os.path.join(UPLOAD_DIR, scan_id)
    os.makedirs(scan_dir, exist_ok=True)
    
    file_path = os.path.join(scan_dir, file.filename)
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    return file_path

def load_scan_data(file_path):
    """Load scan data from a file"""
    if file_path.endswith('.npz'):
        # For NPZ files
        scan_data = np.load(file_path)["arr_0"]
    else:
        # For NIFTI files
        nifti_img = nib.load(file_path)
        scan_data = nifti_img.get_fdata()
    
    return scan_data

def save_metadata(scan_id, metadata):
    """Save metadata for a scan"""
    metadata_path = os.path.join(UPLOAD_DIR, scan_id, "metadata.json")
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

def load_metadata(scan_id):
    """Load metadata for a scan"""
    metadata_path = os.path.join(UPLOAD_DIR, scan_id, "metadata.json")
    
    if not os.path.exists(metadata_path):
        return None
    
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata for scan {scan_id}: {e}")
        return None

def update_scan_status(scan_id, status, error_message=None):
    """Update the status of a scan in its metadata"""
    try:
        metadata = load_metadata(scan_id)
        
        if metadata:
            metadata["status"] = status
            
            if error_message:
                metadata["error_message"] = error_message
            
            save_metadata(scan_id, metadata)
            
    except Exception as e:
        logger.error(f"Error updating scan status: {e}")