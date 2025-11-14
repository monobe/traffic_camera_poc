#!/usr/bin/env python3
"""
Vehicle Type Classifier

Provides detailed vehicle classification beyond YOLO base classes:
- Car -> Light car (kei-car), Regular car, Taxi
- Bus -> Regular bus, Mini bus
- Truck -> Light truck, Regular truck
- Bicycle -> Regular bicycle, E-bike
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class VehicleSubtype:
    """Detailed vehicle subtype"""
    base_type: str  # car, bus, truck, motorcycle, bicycle
    subtype: str    # kei_car, taxi, regular_car, etc.
    confidence: float


class VehicleClassifier:
    """
    Detailed vehicle type classifier

    Uses size, color, and shape features to classify vehicles into subtypes
    """

    # Size thresholds (bbox area relative to image)
    KEI_CAR_MAX_RATIO = 0.03  # Kei cars are small
    REGULAR_CAR_MIN_RATIO = 0.02
    BUS_MIN_RATIO = 0.06  # Buses are large

    # Color ranges for taxi detection (HSV)
    TAXI_YELLOW_LOWER = np.array([20, 100, 100])
    TAXI_YELLOW_UPPER = np.array([30, 255, 255])

    # Aspect ratio thresholds
    BUS_MIN_ASPECT = 2.0  # Buses are long
    MOTORCYCLE_MAX_ASPECT = 1.5

    def __init__(self):
        """Initialize classifier"""
        self.image_area = None

    def classify(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        base_type: str
    ) -> VehicleSubtype:
        """
        Classify vehicle into detailed subtype

        Args:
            frame: Input image
            bbox: Bounding box (x1, y1, x2, y2)
            base_type: YOLO base class (car, bus, truck, etc.)

        Returns:
            VehicleSubtype with detailed classification
        """
        if self.image_area is None:
            h, w = frame.shape[:2]
            self.image_area = h * w

        # Extract vehicle region
        x1, y1, x2, y2 = bbox
        vehicle_crop = frame[y1:y2, x1:x2]

        # Calculate features
        bbox_area = (x2 - x1) * (y2 - y1)
        area_ratio = bbox_area / self.image_area
        aspect_ratio = (x2 - x1) / max((y2 - y1), 1)

        # Classify based on base type
        if base_type == 'car':
            return self._classify_car(vehicle_crop, area_ratio, aspect_ratio)
        elif base_type == 'bus':
            return self._classify_bus(vehicle_crop, area_ratio, aspect_ratio)
        elif base_type == 'truck':
            return self._classify_truck(vehicle_crop, area_ratio, aspect_ratio)
        elif base_type == 'motorcycle':
            return self._classify_motorcycle(vehicle_crop, area_ratio)
        elif base_type == 'bicycle':
            return self._classify_bicycle(vehicle_crop, area_ratio)
        else:
            # Unknown type, return as-is
            return VehicleSubtype(
                base_type=base_type,
                subtype=base_type,
                confidence=0.5
            )

    def _classify_car(
        self,
        crop: np.ndarray,
        area_ratio: float,
        aspect_ratio: float
    ) -> VehicleSubtype:
        """Classify car into kei_car, taxi, or regular_car"""

        # Check if taxi (yellow color)
        if self._is_taxi(crop):
            return VehicleSubtype(
                base_type='car',
                subtype='taxi',
                confidence=0.8
            )

        # Check size for kei car
        if area_ratio < self.KEI_CAR_MAX_RATIO:
            return VehicleSubtype(
                base_type='car',
                subtype='kei_car',
                confidence=0.7
            )

        # Default: regular car
        return VehicleSubtype(
            base_type='car',
            subtype='regular_car',
            confidence=0.7
        )

    def _classify_bus(
        self,
        crop: np.ndarray,
        area_ratio: float,
        aspect_ratio: float
    ) -> VehicleSubtype:
        """Classify bus into regular_bus or mini_bus"""

        # Large and long aspect ratio -> regular bus
        if area_ratio > self.BUS_MIN_RATIO and aspect_ratio > self.BUS_MIN_ASPECT:
            return VehicleSubtype(
                base_type='bus',
                subtype='regular_bus',
                confidence=0.8
            )

        # Smaller -> mini bus
        return VehicleSubtype(
            base_type='bus',
            subtype='mini_bus',
            confidence=0.7
        )

    def _classify_truck(
        self,
        crop: np.ndarray,
        area_ratio: float,
        aspect_ratio: float
    ) -> VehicleSubtype:
        """Classify truck into light_truck or regular_truck"""

        # Small size -> light truck (kei truck)
        if area_ratio < self.KEI_CAR_MAX_RATIO:
            return VehicleSubtype(
                base_type='truck',
                subtype='light_truck',
                confidence=0.7
            )

        # Larger -> regular truck
        return VehicleSubtype(
            base_type='truck',
            subtype='regular_truck',
            confidence=0.7
        )

    def _classify_motorcycle(
        self,
        crop: np.ndarray,
        area_ratio: float
    ) -> VehicleSubtype:
        """Classify motorcycle (currently no subtypes)"""
        return VehicleSubtype(
            base_type='motorcycle',
            subtype='motorcycle',
            confidence=0.8
        )

    def _classify_bicycle(
        self,
        crop: np.ndarray,
        area_ratio: float
    ) -> VehicleSubtype:
        """Classify bicycle (regular vs e-bike - placeholder)"""
        # Note: E-bike detection would require more sophisticated analysis
        # For now, just classify as regular bicycle
        return VehicleSubtype(
            base_type='bicycle',
            subtype='bicycle',
            confidence=0.8
        )

    def _is_taxi(self, crop: np.ndarray) -> bool:
        """
        Detect if vehicle is a taxi based on yellow color

        Args:
            crop: Cropped vehicle image

        Returns:
            True if likely a taxi
        """
        if crop.size == 0:
            return False

        try:
            # Convert to HSV
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            # Create mask for yellow color
            mask = cv2.inRange(hsv, self.TAXI_YELLOW_LOWER, self.TAXI_YELLOW_UPPER)

            # Calculate percentage of yellow pixels
            yellow_ratio = np.count_nonzero(mask) / mask.size

            # If > 20% yellow, likely a taxi
            return yellow_ratio > 0.2

        except Exception:
            return False

    def get_display_name(self, subtype: str, language: str = 'ja') -> str:
        """
        Get display name for subtype

        Args:
            subtype: Vehicle subtype
            language: 'ja' for Japanese, 'en' for English

        Returns:
            Display name string
        """
        if language == 'ja':
            names = {
                'kei_car': '軽自動車',
                'regular_car': '普通車',
                'taxi': 'タクシー',
                'regular_bus': '路線バス',
                'mini_bus': 'マイクロバス',
                'light_truck': '軽トラック',
                'regular_truck': 'トラック',
                'motorcycle': 'バイク',
                'bicycle': '自転車',
                'person': '歩行者'
            }
        else:
            names = {
                'kei_car': 'Kei Car',
                'regular_car': 'Car',
                'taxi': 'Taxi',
                'regular_bus': 'Bus',
                'mini_bus': 'Mini Bus',
                'light_truck': 'Light Truck',
                'regular_truck': 'Truck',
                'motorcycle': 'Motorcycle',
                'bicycle': 'Bicycle',
                'person': 'Pedestrian'
            }

        return names.get(subtype, subtype)
