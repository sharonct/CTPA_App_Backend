import torch
from utils.logging import logger
from models.ct_clip_loader import get_ctclip_model
from config import DEVICE

class SimpleVisionFeatureExtractor:
    """Feature extractor for CT scans using the CT-CLIP model"""
    
    def __init__(self, ctclip_model=None, device=DEVICE):
        self.device = device
        self.ctclip_model = ctclip_model if ctclip_model is not None else get_ctclip_model()
    
    def __call__(self, image_tensor):
        """
        Extract visual features from a CT scan tensor
        
        Args:
            image_tensor: A tensor of shape [1, H, W, D] representing the CT scan
            
        Returns:
            A tensor of visual features
        """
        try:
            # Ensure the model is in eval mode
            self.ctclip_model.eval()
            
            with torch.no_grad():
                # Extract features using the CT-CLIP model's image encoder
                visual_features = self.ctclip_model.encode_image(image_tensor)
                
            return visual_features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise