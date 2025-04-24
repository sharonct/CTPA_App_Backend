import os
import json
import numpy as np
from datetime import datetime
import logging
from config import SCAN_DIR
import uuid

logger = logging.getLogger(__name__)

def save_uploaded_scan(file, filename=None):
    """
    Save an uploaded scan file
    
    Args:
        file: The uploaded file
        filename: Optional filename to use
        
    Returns:
        dict: Scan metadata
    """
    try:
        # Generate a unique scan ID
        scan_id = str(uuid.uuid4())
        
        # Use provided filename or get from file
        if filename is None:
            filename = file.filename
        
        # Create directory for scan if it doesn't exist
        scan_path = os.path.join(SCAN_DIR, scan_id)
        os.makedirs(scan_path, exist_ok=True)
        
        # Save the file
        file_path = os.path.join(scan_path, filename)
        with open(file_path, "wb") as f:
            f.write(file.read())
        
        # Create metadata
        metadata = {
            "scan_id": scan_id,
            "filename": filename,
            "upload_time": datetime.now().isoformat(),
            "status": "uploaded",
            "filepath": file_path
        }
        
        # Save metadata
        metadata_path = os.path.join(scan_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error saving uploaded scan: {e}")
        raise

def load_scan_data(scan_id):
    """
    Load scan data from file
    
    Args:
        scan_id: ID of the scan
        
    Returns:
        numpy.ndarray: Scan data
    """
    try:
        # Get metadata to find the file path
        metadata = load_metadata(scan_id)
        if metadata is None or "filepath" not in metadata:
            raise ValueError(f"Invalid scan ID or missing filepath: {scan_id}")
        
        file_path = metadata["filepath"]
        
        # Check file type and load accordingly
        if file_path.endswith(".npz"):
            # Load NumPy array from .npz file
            data = np.load(file_path)
            return data["arr_0"] if "arr_0" in data else data[list(data.keys())[0]]
        
        elif file_path.endswith(".nii") or file_path.endswith(".nii.gz"):
            # Load NIfTI file using nibabel
            try:
                import nibabel as nib
                nii_img = nib.load(file_path)
                return nii_img.get_fdata()
            except ImportError:
                raise ImportError("nibabel library required to load NIfTI files")
        
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    
    except Exception as e:
        logger.error(f"Error loading scan data: {e}")
        raise

def load_metadata(scan_id):
    """
    Load metadata for a scan
    
    Args:
        scan_id: ID of the scan
        
    Returns:
        dict: Scan metadata
    """
    try:
        metadata_path = os.path.join(SCAN_DIR, scan_id, "metadata.json")
        
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata not found for scan {scan_id}")
            return None
        
        with open(metadata_path, "r") as f:
            return json.load(f)
    
    except Exception as e:
        logger.error(f"Error loading metadata for scan {scan_id}: {e}")
        return None

def save_metadata(scan_id, metadata):
    """
    Save metadata for a scan
    
    Args:
        scan_id: ID of the scan
        metadata: Updated metadata
    """
    try:
        metadata_path = os.path.join(SCAN_DIR, scan_id, "metadata.json")
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving metadata for scan {scan_id}: {e}")
        return False

def update_scan_status(scan_id, status, error_message=None):
    """
    Update the status of a scan
    
    Args:
        scan_id: ID of the scan
        status: New status
        error_message: Optional error message
    """
    try:
        metadata = load_metadata(scan_id)
        
        if metadata:
            metadata["status"] = status
            
            if error_message:
                metadata["error_message"] = error_message
            
            save_metadata(scan_id, metadata)
            
            return True
        
        return False
    
    except Exception as e:
        logger.error(f"Error updating scan status: {e}")
        return False

def get_all_scans():
    """
    Get list of all available scans
    
    Returns:
        list: List of scan metadata
    """
    try:
        scans = []
        
        # Iterate through directories in scan dir
        for scan_id in os.listdir(SCAN_DIR):
            scan_dir = os.path.join(SCAN_DIR, scan_id)
            
            # Skip if not a directory
            if not os.path.isdir(scan_dir):
                continue
            
            # Load metadata
            metadata = load_metadata(scan_id)
            
            if metadata:
                scans.append(metadata)
        
        return scans
    
    except Exception as e:
        logger.error(f"Error getting scan list: {e}")
        return []