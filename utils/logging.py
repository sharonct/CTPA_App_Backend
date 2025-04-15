import logging

def setup_logger():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("api.log")
        ]
    )
    return logging.getLogger(__name__)

# Create logger instance
logger = setup_logger()