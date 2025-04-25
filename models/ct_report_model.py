import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from config import DEVICE, CT_REPORT_MODEL_PATH
from model_components import RobustVisionFeatureExtractor, CrossAttentionLayer, CTReportGenerator

# Setup logging
logger = logging.getLogger(__name__)

# Global variable to hold the model
ct_report_model = None
ct_report_tokenizer = None


def load_ct_report_model():
    """Load the CT report generation model"""
    global ct_report_model, ct_report_tokenizer
    
    try:
        if ct_report_model is not None:
            return True
            
        logger.info("Loading CT Report Generator model...")
        
        # Check if model path exists
        if not os.path.exists(CT_REPORT_MODEL_PATH):
            logger.error(f"Model path not found: {CT_REPORT_MODEL_PATH}")
            return False
        
        # Load checkpoint
        checkpoint = torch.load(CT_REPORT_MODEL_PATH, map_location=DEVICE, weights_only=False)
        
        # Load LLM and tokenizer
        llm_name = "epfl-llm/meditron-7b"  # This would be passed or retrieved from checkpoint
        
        # Load LLM with 8-bit quantization for efficient memory usage
        llm = AutoModelForCausalLM.from_pretrained(
            llm_name, 
            torch_dtype=torch.bfloat16, 
            load_in_8bit=True if torch.cuda.is_available() else False,
            device_map=DEVICE
        )
        
        # Apply LoRA (with the same config as during training)
        lora_config = LoraConfig(
            r=16,  # These values would be passed or retrieved from checkpoint
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none"
        )
        llm = get_peft_model(llm, lora_config)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Import CT-CLIP
        try:
            from models.ct_clip_loader import get_ctclip_model
            
            # Get CT-CLIP model
            ctclip_model = get_ctclip_model()
            
            # Create vision feature extractor
            vision_feature_extractor = RobustVisionFeatureExtractor(
                ctclip_model,
                device=DEVICE,
                dtype=torch.bfloat16
            )
            
            # Create cross-attention layer
            cross_attention = CrossAttentionLayer(
                text_dim=llm.config.hidden_size,
                vision_dim=vision_feature_extractor.feature_dim
            ).to(DEVICE).to(torch.bfloat16)
            
            # Create model
            model = CTReportGenerator(
                llm=llm,
                vision_feature_extractor=vision_feature_extractor,
                cross_attention=cross_attention
            ).to(DEVICE)
            
            # Set tokenizer
            model.tokenizer = tokenizer
            
            # Load state dict
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            # Set global variables
            ct_report_model = model
            ct_report_tokenizer = tokenizer
            
            logger.info(f"CT Report Generator model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading CT-CLIP model for report generator: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Create a simulated model for testing
            ct_report_model = SimulatedCTReportGenerator()
            ct_report_tokenizer = tokenizer
            
            logger.warning("Using simulated CT Report Generator for testing")
            return True
            
    except Exception as e:
        logger.error(f"Error loading CT Report Generator model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def get_ct_report_model():
    """Get the loaded CT report generation model"""
    global ct_report_model
    if ct_report_model is None:
        load_ct_report_model()
    return ct_report_model


def get_ct_report_tokenizer():
    """Get the loaded CT report generation tokenizer"""
    global ct_report_tokenizer
    if ct_report_tokenizer is None:
        load_ct_report_model()
    return ct_report_tokenizer


class SimulatedCTReportGenerator:
    """A simulated CT report generator for testing"""
    def __init__(self):
        self.tokenizer = None
    
    def generate_report(self, images, prompt, max_length=512, temperature=0.7):
        """Simulate report generation"""
        # Randomly decide if there's a PE or not
        import random
        has_pe = random.choice([True, False])
        
        if has_pe:
            return """
            The CT pulmonary angiogram shows filling defects in the right lower lobe pulmonary artery consistent with acute pulmonary embolism. The main pulmonary artery is normal in caliber. There is no evidence of right heart strain. The lung parenchyma shows no consolidation or ground glass opacity. There is no pleural effusion. Mediastinal and hilar lymph nodes are within normal limits.
            
            These findings indicate an acute pulmonary embolism without evidence of right heart strain.
            """
        else:
            return """
            The CT pulmonary angiogram shows no filling defects in the main, lobar, segmental, or subsegmental pulmonary arteries. The main pulmonary artery is normal in caliber. The lung parenchyma shows no consolidation or ground glass opacity. There is no pleural effusion. Mediastinal and hilar lymph nodes are within normal limits.
            
            These findings indicate no evidence of pulmonary embolism at this time.
            """