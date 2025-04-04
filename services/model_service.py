import pickle
import time
import torch
from config import logger, MODEL_PATH, CACHE_EXPIRY_TIME

# Model cache
MODEL_CACHE = {}
LAST_MODEL_USE = {}

def get_model():
    """Load model from disk or cache"""
    try:
        # Check if model is already in cache
        if "model" in MODEL_CACHE:
            # Update last use time
            LAST_MODEL_USE["model"] = time.time()
            return MODEL_CACHE["model"]
        
        # Otherwise load from disk
        logger.info(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            if hasattr(model, 'to') and callable(getattr(model, 'to')):
                model = model.to('cuda')
                model.eval()
        
        # Store in cache
        MODEL_CACHE["model"] = model
        LAST_MODEL_USE["model"] = time.time()
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def cleanup_cache():
    """Clean up model cache if models haven't been used recently"""
    current_time = time.time()
    models_to_remove = []
    
    for model_name, last_use in LAST_MODEL_USE.items():
        if current_time - last_use > CACHE_EXPIRY_TIME:
            models_to_remove.append(model_name)
    
    for model_name in models_to_remove:
        if model_name in MODEL_CACHE:
            logger.info(f"Removing model {model_name} from cache due to inactivity")
            del MODEL_CACHE[model_name]
            del LAST_MODEL_USE[model_name]
    
    # Force garbage collection and GPU cache cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def clear_model_cache():
    """Clear the model cache completely"""
    MODEL_CACHE.clear()
    LAST_MODEL_USE.clear()
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()