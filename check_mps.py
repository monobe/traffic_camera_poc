import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_mps():
    if not torch.backends.mps.is_available():
        logger.info("MPS not available")
        return False
        
    if not torch.backends.mps.is_built():
        logger.info("MPS not built")
        return False
        
    logger.info("MPS is available and built!")
    
    try:
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        logger.info(f"Successfully created tensor on {device}")
        return True
    except Exception as e:
        logger.error(f"Failed to use MPS: {e}")
        return False

if __name__ == "__main__":
    check_mps()
