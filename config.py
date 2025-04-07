import os
import torch

# Application constants
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
CTCLIP_PATH = os.environ.get("CTCLIP_PATH", "/teamspace/studios/this_studio/CT-CLIP_v2.pt")
VQA_MODEL_PATH = os.environ.get("VQA_MODEL_PATH", "/teamspace/studios/this_studio/vqa/model/checkpoint_epoch_10/model_checkpoint.pth")
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "microsoft/BiomedVLP-CXR-BERT-specialized")

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")