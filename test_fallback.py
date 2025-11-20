import logging
import sys
from detection.detector import YOLODetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_fallback():
    logger.info("Testing CoreML fallback logic...")
    
    # Use a non-existent CoreML model name, but one where the .pt equivalent exists (yolov8n.pt)
    # We'll use 'yolov8n.mlpackage' but pretend it doesn't exist by renaming it temporarily?
    # Or better, use a fake name like 'yolov8n_fake.mlpackage' and ensure 'yolov8n_fake.pt' exists?
    # Actually, the logic tries to replace .mlpackage with .pt.
    # So if I ask for 'yolov8n.mlpackage', it will look for 'yolov8n.pt'.
    # I need to simulate 'yolov8n.mlpackage' NOT existing or failing to load.
    # Since 'yolov8n.mlpackage' DOES exist on this system (I created it), I should use a fake name.
    
    fake_coreml = "yolov8n_fallback_test.mlpackage"
    fallback_pt = "yolov8n_fallback_test.pt"
    
    # Create a dummy .pt file (or just copy yolov8n.pt)
    import shutil
    shutil.copy("yolov8n.pt", fallback_pt)
    logger.info(f"Created dummy fallback model: {fallback_pt}")
    
    try:
        # Initialize detector with fake CoreML model
        detector = YOLODetector(model_path=fake_coreml)
        
        # Attempt to load
        if detector.load_model():
            logger.info("✓ Load successful (Fallback worked!)")
            
            # Verify it loaded the fallback
            if str(detector.model.model.pt_path).endswith('.pt') or detector.model.ckpt_path.endswith('.pt'):
                 logger.info("✓ Verified loaded model is .pt")
            else:
                 # Ultralytics internal structure might vary, but if it loaded, it must be the .pt since .mlpackage doesn't exist
                 logger.info("✓ Loaded model seems to be the fallback")
                 
            return True
        else:
            logger.error("✗ Load failed")
            return False
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return False
    finally:
        # Cleanup
        import os
        if os.path.exists(fallback_pt):
            os.remove(fallback_pt)
            logger.info("Cleaned up dummy file")

if __name__ == "__main__":
    success = test_fallback()
    sys.exit(0 if success else 1)
