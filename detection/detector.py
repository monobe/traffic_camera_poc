"""
Object detection module using YOLOv8

Detects vehicles (cars, motorcycles, buses, bicycles) and pedestrians
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Object detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    timestamp: datetime
    subtype: Optional[str] = None  # Detailed vehicle subtype (e.g., 'taxi', 'kei_car')
    subtype_confidence: Optional[float] = None

    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def width(self) -> int:
        """Get width of bounding box"""
        return self.bbox[2] - self.bbox[0]

    def height(self) -> int:
        """Get height of bounding box"""
        return self.bbox[3] - self.bbox[1]

    def area(self) -> int:
        """Get area of bounding box"""
        return self.width() * self.height()


class YOLODetector:
    """YOLOv8 object detector for traffic monitoring"""

    # COCO class IDs for vehicle detection
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        1: 'bicycle',  # Added bicycle
        0: 'person'    # Added pedestrian
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        device: str = "cpu",
        classes: Optional[List[int]] = None,
        imgsz: int = 640,
        enable_classification: bool = True
    ):
        """
        Initialize YOLO detector

        Args:
            model_path: Path to YOLO model file
            confidence: Confidence threshold (0-1)
            device: Device to run on ("cpu" or "cuda")
            classes: List of class IDs to detect (None = all vehicle classes)
            imgsz: Input image size
            enable_classification: Enable detailed vehicle classification
        """
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.imgsz = imgsz
        self.enable_classification = enable_classification

        # Set classes to detect
        if classes is None:
            self.classes = list(self.VEHICLE_CLASSES.keys())
        else:
            self.classes = classes

        self.model = None
        self.is_loaded = False

        # Initialize vehicle classifier
        self.classifier = None
        if enable_classification:
            from .classifier import VehicleClassifier
            self.classifier = VehicleClassifier()

        logger.info(f"YOLODetector initialized:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Confidence: {confidence}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Classes: {self.classes}")
        logger.info(f"  Classification: {'enabled' if enable_classification else 'disabled'}")

    def load_model(self) -> bool:
        """
        Load YOLO model

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model from {self.model_path}...")

            # Check if model file exists
            if not Path(self.model_path).exists():
                logger.info(f"Model file not found, downloading {self.model_path}...")

            self.model = YOLO(self.model_path)
            self.is_loaded = True

            logger.info("✓ YOLO model loaded successfully")
            return True

        except ImportError as e:
            logger.error(f"Failed to import ultralytics: {e}")
            logger.error("Install with: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of Detection objects
        """
        if not self.is_loaded:
            if not self.load_model():
                return []

        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence,
                classes=self.classes,
                device=self.device,
                imgsz=self.imgsz,
                verbose=False
            )

            # Parse results
            detections = []
            timestamp = datetime.now()

            for result in results:
                boxes = result.boxes

                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = (int(x1), int(y1), int(x2), int(y2))

                    # Get class and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Get class name
                    class_name = self.VEHICLE_CLASSES.get(class_id, f"class_{class_id}")

                    # Perform detailed classification if enabled
                    subtype = None
                    subtype_confidence = None
                    if self.classifier:
                        try:
                            vehicle_subtype = self.classifier.classify(frame, bbox, class_name)
                            subtype = vehicle_subtype.subtype
                            subtype_confidence = vehicle_subtype.confidence
                        except Exception as e:
                            logger.debug(f"Classification failed for {class_name}: {e}")

                    # Create detection object
                    detection = Detection(
                        bbox=bbox,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        timestamp=timestamp,
                        subtype=subtype,
                        subtype_confidence=subtype_confidence
                    )

                    detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []

    def detect_and_visualize(
        self,
        frame: np.ndarray,
        show_labels: bool = True,
        show_confidence: bool = True
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Detect objects and draw bounding boxes on frame

        Args:
            frame: Input frame
            show_labels: Show class labels
            show_confidence: Show confidence scores

        Returns:
            Tuple of (annotated frame, detections)
        """
        import cv2

        detections = self.detect(frame)
        annotated_frame = frame.copy()

        # Define colors for each class
        colors = {
            'car': (0, 255, 0),        # Green
            'motorcycle': (255, 0, 0),  # Blue
            'bus': (0, 165, 255),      # Orange
            'truck': (0, 255, 255),    # Yellow
            'bicycle': (255, 255, 0),  # Cyan
            'person': (255, 0, 255)    # Magenta
        }

        for det in detections:
            # Get color for this class
            color = colors.get(det.class_name, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(
                annotated_frame,
                (det.bbox[0], det.bbox[1]),
                (det.bbox[2], det.bbox[3]),
                color,
                2
            )

            # Draw label
            if show_labels:
                # Use subtype if available, otherwise use base class name
                if det.subtype and self.classifier:
                    label = self.classifier.get_display_name(det.subtype, language='ja')
                else:
                    label = det.class_name

                if show_confidence:
                    label += f" {det.confidence:.2f}"

                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    1
                )

                cv2.rectangle(
                    annotated_frame,
                    (det.bbox[0], det.bbox[1] - label_height - 10),
                    (det.bbox[0] + label_width, det.bbox[1]),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (det.bbox[0], det.bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1
                )

        return annotated_frame, detections

    def get_model_info(self) -> dict:
        """
        Get information about loaded model

        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {}

        return {
            'model_path': self.model_path,
            'confidence': self.confidence,
            'device': self.device,
            'classes': self.classes,
            'imgsz': self.imgsz,
            'is_loaded': self.is_loaded
        }
