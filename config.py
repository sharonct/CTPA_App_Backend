import os
import torch

# Application constants
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")

# Model paths
CTCLIP_PATH = os.environ.get("CTCLIP_PATH", "/mnt/c/Users/STRATHMORE/Desktop/Sharon_Tonui/CTPA-CLIP/CTPA_CLIP/models/CT-CLIP_v2.pt")
VQA_MODEL_PATH = os.environ.get("VQA_MODEL_PATH", "/teamspace/studios/this_studio/models/vqa/model/checkpoint_epoch_9/full_model_checkpoint.pth")
CT_REPORT_MODEL_PATH = os.environ.get("CT_REPORT_MODEL_PATH", "/mnt/c/Users/STRATHMORE/Desktop/Sharon_Tonui/CTPA-CLIP/CTPA_CLIP/models/" \
"ct_report/best_model_by_validation.pt")

# Tokenizer and model names
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "microsoft/BiomedVLP-CXR-BERT-specialized")
LLM_NAME = os.environ.get("LLM_NAME", "epfl-llm/meditron-7b")

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(os.path.join(MODEL_DIR, "ct_report", "placeholder")), exist_ok=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Allow simulated mode as fallback
ALLOW_SIMULATED_MODELS = os.environ.get("ALLOW_SIMULATED_MODELS", "True").lower() == "true"

ENABLE_CORS = True  # or False depending on your needs
DEBUG_MODE = True  # or False
USE_VQA_MODEL = False  # or False
USE_CT_REPORT_MODEL = True  # or False
