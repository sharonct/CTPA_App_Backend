import os
import uuid
import torch
import numpy as np
from datetime import datetime
from utils.logging import logger
from utils.file_utils import save_uploaded_file, load_scan_data, save_metadata, update_scan_status
from config import UPLOAD_DIR, DEVICE
from api.schemas import ScanMetadata

async def upload_scan(file, background_tasks=None):
    """
    Upload a CT scan file and create metadata
    
    Args:
        file: The uploaded file
        background_tasks: FastAPI BackgroundTasks for async processing
        
    Returns:
        ScanMetadata object
    """
    # Validate file type
    if not file.filename.lower().endswith(('.nii', '.nii.gz', '.npz')):
        raise ValueError("Invalid file type. Please upload a .nii, .nii.gz, or .npz file")
    
    # Generate a unique ID for the scan
    scan_id = str(uuid.uuid4())
    
    try:
        # Save the uploaded file
        file_path = save_uploaded_file(scan_id, file)
        
        # Create metadata
        metadata = {
            "scan_id": scan_id,
            "filename": file.filename,
            "upload_time": datetime.now().isoformat(),
            "status": "uploaded",
            "filepath": file_path
        }
        
        # Save metadata
        save_metadata(scan_id, metadata)
        
        # Add background task to preprocess scan if needed
        if background_tasks:
            background_tasks.add_task(preprocess_scan, scan_id, file_path)
        
        return ScanMetadata(
            scan_id=scan_id,
            filename=file.filename,
            upload_time=metadata["upload_time"],
            status="uploaded"
        )
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise

def preprocess_scan(scan_id, file_path):
    """
    Preprocess the scan and update metadata
    
    Args:
        scan_id: ID of the scan
        file_path: Path to the scan file
    """
    try:
        # Update status
        update_scan_status(scan_id, "processing")
        
        # Load the scan
        scan_data = load_scan_data(file_path)
        
        # Get dimensions
        dimensions = list(scan_data.shape)
        
        # Update metadata with dimensions and status
        metadata_path = os.path.join(UPLOAD_DIR, scan_id, "metadata.json")
        with open(metadata_path, "r") as f:
            import json
            metadata = json.load(f)
        
        metadata["dimensions"] = dimensions
        metadata["status"] = "ready"
        metadata["preprocessed_time"] = datetime.now().isoformat()
        
        save_metadata(scan_id, metadata)
            
    except Exception as e:
        logger.error(f"Error preprocessing scan {scan_id}: {e}")
        update_scan_status(scan_id, "error", error_message=str(e))

def get_slice(scan_id, view="axial", slice_idx=0, window_center=-600, window_width=1500):
    """
    Get a specific slice from a scan
    
    Args:
        scan_id: ID of the scan
        view: View type (axial, sagittal, or coronal)
        slice_idx: Slice index
        window_center: Window center (HU)
        window_width: Window width (HU)
        
    Returns:
        Dict with slice information and base64 image
    """
    from utils.file_utils import load_metadata
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    
    metadata = load_metadata(scan_id)
    if not metadata:
        raise ValueError(f"Scan with ID {scan_id} not found")
    
    file_path = metadata["filepath"]
    scan_data = load_scan_data(file_path)
    
    # Get dimensions
    dims = scan_data.shape
    
    # Get the slice based on view
    if view == "axial":
        if slice_idx >= dims[2]:
            raise ValueError(f"Slice index out of range. Max: {dims[2]-1}")
        slice_img = scan_data[:, :, slice_idx].T
    elif view == "sagittal":
        if slice_idx >= dims[0]:
            raise ValueError(f"Slice index out of range. Max: {dims[0]-1}")
        slice_img = scan_data[slice_idx, :, :].T
    elif view == "coronal":
        if slice_idx >= dims[1]:
            raise ValueError(f"Slice index out of range. Max: {dims[1]-1}")
        slice_img = scan_data[:, slice_idx, :].T
    else:
        raise ValueError("Invalid view. Must be axial, sagittal, or coronal")
    
    # Apply windowing
    img = slice_img.copy()
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    img[img < min_value] = min_value
    img[img > max_value] = max_value
    img = (img - min_value) / (max_value - min_value) * 255
    
    # Convert to uint8
    img = img.astype(np.uint8)
    
    # Convert to PNG
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='bone')
    ax.axis('off')
    
    # Save to BytesIO
    buf = BytesIO()
    FigureCanvas(fig).print_png(buf)
    plt.close(fig)
    
    # Encode as base64
    img_str = base64.b64encode(buf.getvalue()).decode()
    
    return {
        "scan_id": scan_id,
        "view": view,
        "slice_idx": slice_idx,
        "window_center": window_center,
        "window_width": window_width,
        "dimensions": dims,
        "image": f"data:image/png;base64,{img_str}"
    }