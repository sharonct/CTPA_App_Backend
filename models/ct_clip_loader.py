import torch
from models.ct_clip.ctvit import CTViT
from models.ct_clip.ct_clip import CTCLIP
from transformers import BertTokenizer, BertModel
from utils.logging import logger
from config import CTCLIP_PATH, TOKENIZER_NAME, DEVICE

# Global variables to hold models
ctclip_model = None
tokenizer = None
text_encoder = None
model_loaded = False

def load_ctclip_model():
    """Load CT-CLIP model and related components"""
    global ctclip_model, tokenizer, text_encoder, model_loaded
    
    try:
        if model_loaded:
            return True
            
        logger.info("Loading CT-CLIP model and components...")
        
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME, do_lower_case=True)
        logger.info(f"Loaded tokenizer: {TOKENIZER_NAME}")
        
        # Load text encoder
        text_encoder = BertModel.from_pretrained(TOKENIZER_NAME)
        logger.info("Loaded text encoder")
        
        # Initialize the CT-ViT vision encoder
        image_encoder = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8
        )
        logger.info("Initialized CT-ViT encoder")
        
        # Create the CTCLIP model
        ctclip_model = CTCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            dim_text=768,
            dim_image=294912,
            dim_latent=512,
            extra_latent_projection=False,
            use_mlm=False,
            downsample_image_embeds=False,
            use_all_token_embeds=False
        )
        
        # Load the pre-trained weights
        ctclip_model.load(CTCLIP_PATH)
        ctclip_model = ctclip_model.to(DEVICE)
        logger.info("Loaded CT-CLIP model")
        
        model_loaded = True
        return True
        
    except Exception as e:
        logger.error(f"Error loading CT-CLIP model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def get_ctclip_model():
    """Get the loaded CT-CLIP model"""
    global ctclip_model
    if ctclip_model is None:
        load_ctclip_model()
    return ctclip_model

def get_tokenizer():
    """Get the loaded tokenizer"""
    global tokenizer
    if tokenizer is None:
        load_ctclip_model()
    return tokenizer

def get_text_encoder():
    """Get the loaded text encoder"""
    global text_encoder
    if text_encoder is None:
        load_ctclip_model()
    return text_encoder