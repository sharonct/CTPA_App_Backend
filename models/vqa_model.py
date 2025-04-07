import torch
from transformers import BertModel
from utils.logging import logger
from models.ct_clip_loader import get_text_encoder, get_tokenizer
from models.feature_extractor import SimpleVisionFeatureExtractor
from direct_replacement_code import MedicalVQAModel
from config import VQA_MODEL_PATH, DEVICE

# Global variable to hold the VQA model
vqa_model = None
feature_extractor = None

def load_vqa_model():
    """Load the VQA model"""
    global vqa_model, feature_extractor
    
    try:
        if vqa_model is not None:
            return vqa_model
            
        logger.info("Loading VQA model...")
        
        # Get text encoder and tokenizer
        text_encoder = get_text_encoder()
        tokenizer = get_tokenizer()
        
        # Initialize VQA model
        vqa_model = MedicalVQAModel(
            text_encoder=text_encoder,
            vision_feature_dim=512,
            text_feature_dim=768,
            vocab_size=tokenizer.vocab_size
        ).to(DEVICE)
        
        # Load VQA model weights
        checkpoint = torch.load(VQA_MODEL_PATH, map_location=DEVICE)
        vqa_model.load_state_dict(checkpoint['model_state_dict'])
        vqa_model.eval()
        
        # Initialize feature extractor
        feature_extractor = SimpleVisionFeatureExtractor()
        logger.info("Loaded VQA model and feature extractor")
        
        return vqa_model
        
    except Exception as e:
        logger.error(f"Error loading VQA model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def get_vqa_model():
    """Get the loaded VQA model"""
    global vqa_model
    if vqa_model is None:
        load_vqa_model()
    return vqa_model

def get_feature_extractor():
    """Get the loaded feature extractor"""
    global feature_extractor
    if feature_extractor is None:
        load_vqa_model()
    return feature_extractor