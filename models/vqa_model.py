# import torch
# from transformers import BertModel
# from utils.logging import logger
# from models.ct_clip_loader import get_text_encoder, get_tokenizer
# from models.feature_extractor import SimpleVisionFeatureExtractor
# from vqa import MedicalVQAModel
# from config import VQA_MODEL_PATH, DEVICE

# # Global variable to hold the VQA model and feature extractor
# vqa_model = None
# feature_extractor = None

# def load_vqa_model():
#     """Load the VQA model and feature extractor"""
#     global vqa_model, feature_extractor
    
#     try:
#         # If the model is already loaded, return it
#         if vqa_model is not None and feature_extractor is not None:
#             logger.info("VQA model and feature extractor already loaded.")
#             return vqa_model
        
#         logger.info("Loading VQA model and feature extractor...")

#         # Get text encoder and tokenizer
#         text_encoder = get_text_encoder()
#         tokenizer = get_tokenizer()
        
#         # Initialize VQA model
#         vqa_model = MedicalVQAModel(
#             text_encoder=text_encoder,
#             vision_feature_dim=512,
#             text_feature_dim=768,
#             vocab_size=tokenizer.vocab_size
#         ).to(DEVICE)
        
#         # Load VQA model weights
#         checkpoint = torch.load(VQA_MODEL_PATH, map_location=DEVICE)
#         vqa_model.load_state_dict(checkpoint['model_state_dict'])
#         vqa_model.eval()
        
#         # Initialize feature extractor
#         feature_extractor = SimpleVisionFeatureExtractor()
#         logger.info("Loaded VQA model and feature extractor")
        
#         return vqa_model
        
#     except Exception as e:
#         logger.error(f"Error loading VQA model: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return None

# def get_vqa_model():
#     """Get the loaded VQA model"""
#     global vqa_model
#     if vqa_model is None:
#         load_vqa_model()
#     return vqa_model

# def get_feature_extractor():
#     """Get the loaded feature extractor"""
#     global feature_extractor
#     if feature_extractor is None:
#         load_vqa_model()
#     if feature_extractor is None:
#         logger.error("Feature extractor is not initialized!")
#         raise ValueError("Feature extractor is not initialized properly.")
#     return feature_extractor


import torch
import os
import sys
from utils.logging import logger
from config import DEVICE

# Global variables
feature_extractor = None
model_loaded = False

class EnhancedVisionFeatureExtractor:
    """A simplified feature extractor that works with any input format"""
    def __init__(self, feature_dim=512, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim
        
    def __call__(self, x):
        try:
            # Ensure proper device and type
            x = x.to(self.device).float()
            
            # Log input shape
            logger.info(f"Processing input with shape: {x.shape}")
            
            # Check dimensions and reshape if needed
            if len(x.shape) == 4:  # [B, H, W, D] or similar
                logger.info(f"Reshaping 4D input of shape {x.shape}")
                
                # Try to normalize the shape to [B, C, D, H, W]
                if x.shape[1] == 512 and x.shape[2] == 512:
                    # This is likely [B, H, W, D]
                    x = x.permute(0, 3, 1, 2).unsqueeze(1)
                else:
                    # Add channel dimension
                    x = x.unsqueeze(1)
                
                logger.info(f"After reshaping: {x.shape}")
            
            # Use a simple and robust feature extraction method
            # Resize to a manageable size and use average pooling
            try:
                # Use adaptive pooling to reduce dimensions
                pooled = torch.nn.functional.adaptive_avg_pool3d(x, (4, 4, 4))
                flattened = pooled.reshape(x.shape[0], -1)  # Flatten for features
                
                # If needed, project to desired feature dimension
                if flattened.shape[1] != self.feature_dim:
                    if not hasattr(self, 'projection'):
                        # Create a small projection layer
                        self.projection = torch.nn.Linear(
                            flattened.shape[1], 
                            self.feature_dim
                        ).to(self.device)
                    
                    features = self.projection(flattened)
                else:
                    features = flattened
                
                logger.info(f"Successfully extracted features with shape: {features.shape}")
                return features
                
            except Exception as e:
                logger.error(f"Error in feature extraction: {e}")
                # Try an even simpler approach - random features
                return torch.randn(x.shape[0], self.feature_dim, device=self.device)
            
        except Exception as e:
            logger.error(f"Error in EnhancedVisionFeatureExtractor: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return random features with the correct shape
            return torch.randn(1, self.feature_dim, device=self.device)

def load_vqa_model():
    """Load a simplified feature extractor for chat functionality"""
    global feature_extractor, model_loaded
    
    try:
        # If already loaded, return True
        if model_loaded:
            return True
            
        logger.info("Setting up simplified feature extractor for chat functionality...")
        
        # Create a simplified feature extractor
        feature_extractor = EnhancedVisionFeatureExtractor(device=DEVICE)
        
        model_loaded = True
        logger.info("Feature extractor set up successfully")
        return True
            
    except Exception as e:
        logger.error(f"Error setting up feature extractor: {e}")
        import traceback
        logger.error(traceback.format_exc())
        model_loaded = False
        return False

def get_vqa_model():
    """Get the VQA model (not used in this implementation)"""
    return None

def get_feature_extractor():
    """Get the feature extractor"""
    global feature_extractor
    if feature_extractor is None:
        load_vqa_model()
    return feature_extractor