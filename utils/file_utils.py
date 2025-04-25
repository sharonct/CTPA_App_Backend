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


def load_preprocessed_scan(file_path):
    data = np.load(file_path)
    
    # Assuming the array is saved as the first (and only) item in the .npz file
    scan_array = data[data.files[0]]
    
    # Convert to torch tensor
    # Add batch and channel dimensions
    tensor = torch.from_numpy(scan_array).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, depth, height, width]
    
    return tensor

def preprocess_ct_scan(ct_scan, target_size=480, target_depth=240):
    # Ensure tensor is 4D or 5D: [batch, channel, depth, height, width]
    if ct_scan.dim() == 3:
        ct_scan = ct_scan.unsqueeze(0)
    
    batch, channel, depth, height, width = ct_scan.shape
    
    # Select middle slices and reshape to match model expectations
    if depth > target_depth:
        start = (depth - target_depth) // 2
        ct_scan = ct_scan[:, :, start:start+target_depth, :, :]
    elif depth < target_depth:
        # Pad if fewer slices
        pad_size = target_depth - depth
        pad_before = pad_size // 2
        pad_after = pad_size - pad_before
        ct_scan = F.pad(ct_scan, (0, 0, 0, 0, pad_before, pad_after), mode='constant', value=0)
    
    # Resize to target size (260 allows for 13x20 patches)
    ct_scan_resized = F.interpolate(
        ct_scan.float(), 
        size=(target_depth, target_size, target_size), 
        mode='trilinear', 
        align_corners=False
    )
    
    return ct_scan_resized

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