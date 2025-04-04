import numpy as np
import torch
from config import logger

def preprocess_scan(scan_data):
    """Preprocess scan data for model input"""
    try:
        # Example preprocessing (adjust based on your model requirements)
        # Normalize to 0-1 range
        if np.min(scan_data) != np.max(scan_data):
            scan_data = (scan_data - np.min(scan_data)) / (np.max(scan_data) - np.min(scan_data))
        
        # Add batch and channel dimensions if needed
        processed_data = np.expand_dims(np.expand_dims(scan_data, axis=0), axis=0)
        
        # Convert to PyTorch tensor if using PyTorch
        if torch.cuda.is_available():
            processed_data = torch.from_numpy(processed_data).float().to('cuda')
        else:
            processed_data = torch.from_numpy(processed_data).float()
        
        return processed_data
    except Exception as e:
        logger.error(f"Error preprocessing scan: {str(e)}")
        raise