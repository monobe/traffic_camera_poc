"""
Object detection module using YOLOv8

Detects vehicles (cars, motorcycles, buses, bicycles) and pedestrians
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
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
        enable_classification: bool = True,
        ignore_zones: Optional[List[List[int]]] = None
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
            ignore_zones: List of [x1, y1, x2, y2] zones to ignore detections
        """
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.imgsz = imgsz
        self.enable_classification = enable_classification
        self.ignore_zones = ignore_zones or []

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
        if self.ignore_zones:
            logger.info(f"  Ignore zones: {self.ignore_zones}")

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

            # Filter out detections in ignore zones
            if self.ignore_zones:
                detections = self._filter_ignore_zones(detections)

            return detections

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []

    def _filter_ignore_zones(self, detections: List[Detection]) -> List[Detection]:
        """
        Filter out detections that are in ignore zones

        Args:
            detections: List of detections

        Returns:
            Filtered list of detections
        """
        filtered = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            cx, cy = detection.center()

            # Check if detection center is in any ignore zone
            in_ignore_zone = False
            for zone in self.ignore_zones:
                zx1, zy1, zx2, zy2 = zone
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    in_ignore_zone = True
                    logger.debug(f"Ignored {detection.class_name} at ({cx}, {cy}) in zone {zone}")
                    break

            if not in_ignore_zone:
                filtered.append(detection)

        return filtered

    def _draw_text_pil(
        self,
        img: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_size: int = 20,
        text_color: Tuple[int, int, int] = (0, 0, 0),
        bg_color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Draw Japanese text using PIL (supports Unicode)

        Args:
            img: OpenCV image (BGR)
            text: Text to draw
            position: (x, y) position
            font_size: Font size
            text_color: Text color (BGR)
            bg_color: Background color (BGR), None for transparent

        Returns:
            Image with text drawn
        """
        from PIL import Image, ImageDraw, ImageFont
        import sys

        # Convert BGR to RGB
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Try to load Japanese font
        font = None
        font_paths = [
            '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',  # macOS Hiragino
            '/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Linux
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/Library/Fonts/Arial Unicode.ttf',  # macOS Arial Unicode
            'C:\\Windows\\Fonts\\msgothic.ttc',  # Windows MS Gothic
        ]

        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue

        # Fallback to default font if no Japanese font found
        if font is None:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()

        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw background rectangle if specified
        if bg_color is not None:
            # Convert BGR to RGB
            bg_color_rgb = (bg_color[2], bg_color[1], bg_color[0])
            draw.rectangle(
                [position[0], position[1] - text_height - 5,
                 position[0] + text_width + 5, position[1] + 5],
                fill=bg_color_rgb
            )

        # Draw text (convert BGR to RGB)
        text_color_rgb = (text_color[2], text_color[1], text_color[0])
        draw.text(position, text, font=font, fill=text_color_rgb)

        # Convert back to BGR
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

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

                # Draw label using PIL for Japanese text support
                annotated_frame = self._draw_text_pil(
                    annotated_frame,
                    label,
                    (det.bbox[0] + 3, det.bbox[1] - 25),
                    font_size=18,
                    text_color=(0, 0, 0),  # Black text
                    bg_color=color
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
