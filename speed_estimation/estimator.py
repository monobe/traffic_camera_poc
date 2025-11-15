"""
Speed estimation module

Calculate vehicle speed from tracked trajectories using calibration data
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """Camera calibration data"""
    point1: tuple
    point2: tuple
    real_distance_meters: float
    pixel_distance: float
    pixels_per_meter: float
    calibration_date: str

    @classmethod
    def from_file(cls, filepath: str) -> 'CalibrationData':
        """
        Load calibration data from JSON file

        Args:
            filepath: Path to calibration JSON file

        Returns:
            CalibrationData object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls(
            point1=tuple(data['point1']),
            point2=tuple(data['point2']),
            real_distance_meters=data['real_distance_meters'],
            pixel_distance=data['pixel_distance'],
            pixels_per_meter=data['pixels_per_meter'],
            calibration_date=data['calibration_date']
        )


@dataclass
class SpeedEstimate:
    """Speed estimation result"""
    track_id: int
    object_type: str
    speed_kmh: float
    direction: str  # 'uphill', 'downhill', 'horizontal'
    confidence: float
    timestamp: datetime
    trajectory_length: int
    distance_meters: float
    time_seconds: float
    vehicle_subtype: Optional[str] = None  # Detailed vehicle subtype

    def is_speeding(self, speed_limit_kmh: float) -> bool:
        """Check if speed exceeds limit"""
        return self.speed_kmh > speed_limit_kmh


class SpeedEstimator:
    """Estimate speed from object trajectories"""

    def __init__(
        self,
        calibration_file: str,
        fps: int = 15,
        min_track_length: int = 10,
        speed_limit_kmh: float = 30.0,
        direction_threshold: int = 50
    ):
        """
        Initialize speed estimator

        Args:
            calibration_file: Path to calibration JSON file
            fps: Camera FPS for time calculation
            min_track_length: Minimum trajectory points to estimate speed
            speed_limit_kmh: Speed limit for reporting
            direction_threshold: Pixels of vertical movement to determine direction
        """
        self.fps = fps
        self.min_track_length = min_track_length
        self.speed_limit_kmh = speed_limit_kmh
        self.direction_threshold = direction_threshold

        # Load calibration
        self.calibration = None
        if Path(calibration_file).exists():
            self.calibration = CalibrationData.from_file(calibration_file)
            logger.info(f"Calibration loaded:")
            logger.info(f"  Pixels per meter: {self.calibration.pixels_per_meter:.2f}")
            logger.info(f"  Real distance: {self.calibration.real_distance_meters:.2f}m")
        else:
            logger.warning(f"Calibration file not found: {calibration_file}")
            logger.warning("Speed estimation will not be accurate!")

        logger.info(f"SpeedEstimator initialized:")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Min track length: {min_track_length}")
        logger.info(f"  Speed limit: {speed_limit_kmh} km/h")

    def update_fps(self, fps: float):
        """
        Update FPS value dynamically based on actual measured frame rate

        Args:
            fps: Measured actual FPS
        """
        if fps > 0:
            old_fps = self.fps
            self.fps = fps
            if abs(fps - old_fps) > 1.0:
                logger.info(f"FPS updated: {old_fps:.2f} -> {fps:.2f} FPS")

    def estimate_speed(self, track) -> Optional[SpeedEstimate]:
        """
        Estimate speed from track trajectory

        Args:
            track: Track object with trajectory

        Returns:
            SpeedEstimate object or None if not enough data
        """
        if self.calibration is None:
            logger.warning("No calibration data, cannot estimate speed")
            return None

        # Check if track has enough points
        if len(track.trajectory) < self.min_track_length:
            return None

        # Calculate distance in pixels
        pixel_distance = self._calculate_trajectory_distance(track.trajectory)

        if pixel_distance == 0:
            return None

        # Convert to meters
        distance_meters = pixel_distance / self.calibration.pixels_per_meter

        # Minimum distance filter: only calculate speed if vehicle moved at least 0.5m
        # This prevents calculating speed for stationary or barely-moving objects
        MIN_DISTANCE_METERS = 0.5
        if distance_meters < MIN_DISTANCE_METERS:
            return None

        # Calculate time (number of frames / FPS)
        time_seconds = len(track.trajectory) / self.fps

        if time_seconds == 0:
            return None

        # Calculate speed (m/s)
        speed_ms = distance_meters / time_seconds

        # Convert to km/h
        speed_kmh = speed_ms * 3.6

        # Determine direction
        direction = self._determine_direction(track.trajectory)

        # Calculate confidence (based on track length and average confidence)
        confidence = min(1.0, len(track.trajectory) / (self.min_track_length * 3))
        if hasattr(track, 'get_average_confidence'):
            confidence *= track.get_average_confidence()

        return SpeedEstimate(
            track_id=track.track_id,
            object_type=track.class_name,
            speed_kmh=speed_kmh,
            direction=direction,
            confidence=confidence,
            timestamp=track.timestamps[-1] if track.timestamps else datetime.now(),
            trajectory_length=len(track.trajectory),
            distance_meters=distance_meters,
            time_seconds=time_seconds,
            vehicle_subtype=getattr(track, 'vehicle_subtype', None)
        )

    def _calculate_trajectory_distance(self, trajectory: list) -> float:
        """
        Calculate total distance traveled along trajectory in pixels

        Args:
            trajectory: List of (x, y) points

        Returns:
            Total distance in pixels
        """
        if len(trajectory) < 2:
            return 0.0

        total_distance = 0.0

        for i in range(1, len(trajectory)):
            p1 = trajectory[i-1]
            p2 = trajectory[i]

            # Calculate Euclidean distance
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_distance += distance

        return total_distance

    def _determine_direction(self, trajectory: list) -> str:
        """
        Determine movement direction from trajectory

        Args:
            trajectory: List of (x, y) points

        Returns:
            Direction string: 'uphill', 'downhill', or 'horizontal'
        """
        if len(trajectory) < 2:
            return 'unknown'

        # Calculate vertical movement (y direction)
        start_y = trajectory[0][1]
        end_y = trajectory[-1][1]
        vertical_movement = end_y - start_y

        # Determine direction based on threshold
        if abs(vertical_movement) < self.direction_threshold:
            return 'horizontal'
        elif vertical_movement > 0:
            return 'downhill'  # In image coordinates, y increases downward
        else:
            return 'uphill'

    def is_speed_valid(self, speed_kmh: float, max_speed_kmh: float = 150.0) -> bool:
        """
        Check if estimated speed is valid (not unrealistic)

        Args:
            speed_kmh: Estimated speed
            max_speed_kmh: Maximum realistic speed

        Returns:
            True if speed is valid
        """
        return 0 <= speed_kmh <= max_speed_kmh

    def get_speed_category(self, speed_kmh: float) -> str:
        """
        Categorize speed

        Args:
            speed_kmh: Speed in km/h

        Returns:
            Category string
        """
        if speed_kmh < 20:
            return 'slow'
        elif speed_kmh < self.speed_limit_kmh:
            return 'normal'
        elif speed_kmh < self.speed_limit_kmh + 10:
            return 'slightly_over'
        elif speed_kmh < self.speed_limit_kmh + 20:
            return 'over'
        else:
            return 'significantly_over'

    def format_speed_report(self, estimate: SpeedEstimate) -> str:
        """
        Format speed estimate for logging/reporting

        Args:
            estimate: SpeedEstimate object

        Returns:
            Formatted string
        """
        speeding_indicator = "⚠️ " if estimate.is_speeding(self.speed_limit_kmh) else "✓ "
        category = self.get_speed_category(estimate.speed_kmh)

        report = (
            f"{speeding_indicator}"
            f"Track #{estimate.track_id} "
            f"({estimate.object_type}): "
            f"{estimate.speed_kmh:.1f} km/h "
            f"[{estimate.direction}] "
            f"({category}, "
            f"conf: {estimate.confidence:.2f}, "
            f"dist: {estimate.distance_meters:.1f}m, "
            f"time: {estimate.time_seconds:.1f}s)"
        )

        return report
