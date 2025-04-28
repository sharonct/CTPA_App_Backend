import os
import json
import nibabel as nib
import numpy as np
import shutil
from datetime import datetime
from utils.logging import logger
from config import UPLOAD_DIR
import torch
import torch.nn.functional as F


def save_uploaded_file(scan_id, file):
    """Save an uploaded file to the scan directory"""
    scan_dir = os.path.join(UPLOAD_DIR, scan_id)
    os.makedirs(scan_dir, exist_ok=True)
    
    file_path = os.path.join(scan_dir, file.filename)
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    return file_path

def preprocess_nifti(file_path, metadata=None, target_size=480, target_depth=240):
    """
    Preprocess a NIFTI file for inference, following the same steps as the training pipeline
    
    Args:
        file_path: Path to the NIFTI file (.nii or .nii.gz)
        metadata: Optional dictionary with metadata (slope, intercept, xy_spacing, z_spacing)
        target_size: Target size for height and width
        target_depth: Target size for depth
    
    Returns:
        numpy.ndarray: Preprocessed CT scan data
    """
    try:
        logger.info(f"Loading NIFTI file from {file_path}")
        # Load the NIFTI file
        nii_img = nib.load(file_path)
        img_data = nii_img.get_fdata()
        
        # Default values to use if metadata is not provided
        slope = 1.0
        intercept = 0.0
        xy_spacing = 1.0
        z_spacing = 1.0
        
        # If metadata is provided, extract the relevant values
        if metadata is not None:
            if isinstance(metadata, dict):
                # Extract values from dictionary
                slope = float(metadata.get("RescaleSlope", metadata.get("slope", slope)))
                intercept = float(metadata.get("RescaleIntercept", metadata.get("intercept", intercept)))
                xy_spacing = float(metadata.get("XYSpacing", metadata.get("xy_spacing", xy_spacing)))
                z_spacing = float(metadata.get("ZSpacing", metadata.get("z_spacing", z_spacing)))
        else:
            # Try to extract spacing info from the NIFTI header if available
            try:
                # Get pixel dimensions from NIfTI header
                pixdim = nii_img.header.get('pixdim')
                if pixdim is not None and len(pixdim) >= 4:
                    # pixdim[1:4] contains the x, y, z voxel dimensions
                    xy_spacing = (pixdim[1] + pixdim[2]) / 2  # Average of x and y spacing
                    z_spacing = pixdim[3]
                    logger.info(f"Extracted spacing from NIFTI header: xy={xy_spacing}, z={z_spacing}")
            except Exception as e:
                logger.warning(f"Could not extract spacing from NIFTI header: {e}")
        
        # Apply rescale slope and intercept to get proper Hounsfield Units
        img_data = slope * img_data + intercept
        
        # Clip values to HU range -1000 to 1000
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)
        
        # Normalize by dividing by 1000
        img_data = (img_data / 1000).astype(np.float32)
        
        # Convert to the right orientation - NIFTI data is often [H, W, D]
        # but we expect [D, H, W] for our model
        if img_data.shape[2] < img_data.shape[0] and img_data.shape[2] < img_data.shape[1]:
            # If the third dimension is much smaller, it's likely the depth dimension
            img_data = np.transpose(img_data, (2, 0, 1))
        
        # Convert to tensor for processing
        tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        
        # Process dimensions
        batch, channel, depth, height, width = tensor.shape
        
        # Handle depth dimension
        if depth > target_depth:
            start = (depth - target_depth) // 2
            tensor = tensor[:, :, start:start+target_depth, :, :]
        elif depth < target_depth:
            pad_size = target_depth - depth
            pad_before = pad_size // 2
            pad_after = pad_size - pad_before
            tensor = F.pad(tensor, (0, 0, 0, 0, pad_before, pad_after), mode='constant', value=0)
        
        # Resize to target size
        resized_tensor = F.interpolate(
            tensor.float(), 
            size=(target_depth, target_size, target_size), 
            mode='trilinear', 
            align_corners=False
        )
        
        logger.info(f"Preprocessed NIFTI file to shape: {resized_tensor.shape}")
        return resized_tensor
        
    except Exception as e:
        logger.error(f"Error preprocessing NIFTI file: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


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