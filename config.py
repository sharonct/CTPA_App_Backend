import os
import tempfile
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backend_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("backend_server")

# Model path from environment variable or default
MODEL_PATH = os.environ.get("CTPA_MODEL_PATH", "models/ctpa_model.pkl")

# Model cache settings
CACHE_EXPIRY_TIME = 1800  # 30 minutes

# Create temp directory for uploads
UPLOAD_DIR = tempfile.mkdtemp()

# Get port from environment variable or default to 8000
PORT = int(os.environ.get("PORT", 8000))