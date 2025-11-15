"""
Motion-based detector using background subtraction

Lightweight alternative to YOLO for speed detection.
Uses OpenCV background subtraction to detect moving vehicles.
"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detector import Detection


logger = logging.getLogger(__name__)


class MotionDetector:
    """Lightweight motion-based vehicle detector"""

    def __init__(
        self,
        min_area: int = 2000,
        max_area: int = 100000,
        confidence: float = 0.8,
        learning_rate: float = 0.001,
        ignore_zones: Optional[List[List[int]]] = None
    ):
        """
        Initialize motion detector

        Args:
            min_area: Minimum contour area (pixels) to consider as vehicle
            max_area: Maximum contour area (pixels) to consider as vehicle
            confidence: Fixed confidence score for all detections
            learning_rate: Background subtractor learning rate
            ignore_zones: List of [x1, y1, x2, y2] zones to ignore detections
        """
        self.min_area = min_area
        self.max_area = max_area
        self.confidence = confidence
        self.ignore_zones = ignore_zones or []

        # Create background subtractor (MOG2 algorithm)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        self.learning_rate = learning_rate

        self.is_loaded = True
        self.frame_count = 0

        logger.info(f"MotionDetector initialized:")
        logger.info(f"  Min area: {min_area} pixels")
        logger.info(f"  Max area: {max_area} pixels")
        logger.info(f"  Confidence: {confidence}")
        logger.info(f"  Learning rate: {learning_rate}")
        if self.ignore_zones:
            logger.info(f"  Ignore zones: {self.ignore_zones}")

    def load_model(self) -> bool:
        """
        Load model (compatibility method)

        Returns:
            True (always ready)
        """
        return True

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect moving objects in frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of Detection objects
        """
        self.frame_count += 1

        try:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)

            # Remove shadows (value 127)
            fg_mask[fg_mask == 127] = 0

            # Apply morphological operations to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                fg_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Convert contours to detections
            detections = []
            timestamp = datetime.now()

            for contour in contours:
                area = cv2.contourArea(contour)

                # Filter by area
                if area < self.min_area or area > self.max_area:
                    continue

                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                bbox = (x, y, x + w, y + h)

                # Check aspect ratio (avoid very thin/wide objects)
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                    continue

                # Create detection
                detection = Detection(
                    bbox=bbox,
                    class_id=2,  # Use class ID 2 (car) as default
                    class_name='vehicle',
                    confidence=self.confidence,
                    timestamp=timestamp
                )

                detections.append(detection)

            # Filter out detections in ignore zones
            if self.ignore_zones:
                detections = self._filter_ignore_zones(detections)

            return detections

        except Exception as e:
            logger.error(f"Error during motion detection: {e}")
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
            cx, cy = detection.center()

            # Check if detection center is in any ignore zone
            in_ignore_zone = False
            for zone in self.ignore_zones:
                zx1, zy1, zx2, zy2 = zone
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    in_ignore_zone = True
                    logger.debug(f"Ignored vehicle at ({cx}, {cy}) in zone {zone}")
                    break

            if not in_ignore_zone:
                filtered.append(detection)

        return filtered

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

        # Color for all vehicles (green)
        color = (0, 255, 0)

        for det in detections:
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
                label = "Vehicle"
                if show_confidence:
                    label += f" {det.confidence:.2f}"

                cv2.putText(
                    annotated_frame,
                    label,
                    (det.bbox[0] + 3, det.bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        return annotated_frame, detections

    def get_model_info(self) -> dict:
        """
        Get information about detector

        Returns:
            Dictionary with detector information
        """
        return {
            'detector_type': 'motion',
            'algorithm': 'MOG2',
            'min_area': self.min_area,
            'max_area': self.max_area,
            'confidence': self.confidence,
            'is_loaded': self.is_loaded,
            'frames_processed': self.frame_count
        }
