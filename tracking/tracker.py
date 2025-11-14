"""
Object tracking module using ByteTrack/SORT

Track detected objects across frames for speed estimation
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Object track across multiple frames"""
    track_id: int
    class_name: str
    trajectory: List[Tuple[int, int]] = field(default_factory=list)  # (x, y) center points
    timestamps: List[datetime] = field(default_factory=list)
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    state: str = 'active'  # 'active', 'lost', 'finished'
    frames_since_update: int = 0
    total_frames: int = 0

    def update(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        timestamp: datetime
    ):
        """
        Update track with new detection

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Detection confidence
            timestamp: Detection timestamp
        """
        # Calculate center point
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2

        self.trajectory.append((center_x, center_y))
        self.timestamps.append(timestamp)
        self.bbox_history.append(bbox)
        self.confidence_history.append(confidence)

        self.frames_since_update = 0
        self.total_frames += 1
        self.state = 'active'

    def get_last_position(self) -> Optional[Tuple[int, int]]:
        """Get last known position"""
        if self.trajectory:
            return self.trajectory[-1]
        return None

    def get_last_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get last bounding box"""
        if self.bbox_history:
            return self.bbox_history[-1]
        return None

    def predict_position(self) -> Optional[Tuple[int, int]]:
        """
        Predict next position based on trajectory

        Returns:
            Predicted (x, y) position or None
        """
        if len(self.trajectory) < 2:
            return self.get_last_position()

        # Use last two points to predict
        p1 = self.trajectory[-2]
        p2 = self.trajectory[-1]

        # Calculate velocity
        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]

        # Predict next position
        pred_x = p2[0] + vx
        pred_y = p2[1] + vy

        return (int(pred_x), int(pred_y))

    def get_average_confidence(self) -> float:
        """Get average confidence across all detections"""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

    def get_trajectory_length(self) -> int:
        """Get number of points in trajectory"""
        return len(self.trajectory)

    def mark_lost(self):
        """Mark track as lost"""
        self.state = 'lost'
        self.frames_since_update += 1

    def mark_finished(self):
        """Mark track as finished"""
        self.state = 'finished'


class ObjectTracker:
    """Multi-object tracker using IoU-based matching"""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize tracker

        Args:
            max_age: Maximum frames to keep lost tracks
            min_hits: Minimum detections to confirm a track
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_count = 0

        logger.info(f"ObjectTracker initialized:")
        logger.info(f"  Max age: {max_age}")
        logger.info(f"  Min hits: {min_hits}")
        logger.info(f"  IoU threshold: {iou_threshold}")

    def update(
        self,
        detections: List,
        timestamp: Optional[datetime] = None
    ) -> List[Track]:
        """
        Update tracks with new detections

        Args:
            detections: List of Detection objects
            timestamp: Current timestamp

        Returns:
            List of active tracks
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.frame_count += 1

        # If no detections, increment age for all tracks
        if not detections:
            for track in self.tracks.values():
                track.mark_lost()
            self._remove_old_tracks()
            return self.get_active_tracks()

        # Match detections to existing tracks
        matched_tracks, unmatched_detections = self._match_detections(detections)

        # Update matched tracks
        for track_id, detection in matched_tracks.items():
            self.tracks[track_id].update(
                bbox=detection.bbox,
                confidence=detection.confidence,
                timestamp=timestamp
            )

        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1

            track = Track(
                track_id=track_id,
                class_name=detection.class_name
            )
            track.update(
                bbox=detection.bbox,
                confidence=detection.confidence,
                timestamp=timestamp
            )

            self.tracks[track_id] = track
            logger.debug(f"Created new track {track_id} ({detection.class_name})")

        # Mark unmatched tracks as lost
        matched_track_ids = set(matched_tracks.keys())
        for track_id, track in self.tracks.items():
            if track_id not in matched_track_ids:
                track.mark_lost()

        # Remove old tracks
        self._remove_old_tracks()

        return self.get_active_tracks()

    def _match_detections(
        self,
        detections: List
    ) -> Tuple[Dict[int, any], List]:
        """
        Match detections to existing tracks using IoU

        Args:
            detections: List of Detection objects

        Returns:
            Tuple of (matched_tracks dict, unmatched_detections list)
        """
        if not self.tracks:
            return {}, detections

        # Calculate IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))

        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            last_bbox = track.get_last_bbox()

            if last_bbox is None:
                continue

            for j, detection in enumerate(detections):
                # Only match same class
                if track.class_name != detection.class_name:
                    continue

                iou = self._calculate_iou(last_bbox, detection.bbox)
                iou_matrix[i, j] = iou

        # Match using greedy algorithm (can be improved with Hungarian algorithm)
        matched_tracks = {}
        unmatched_detections = list(detections)

        # Sort matches by IoU (highest first)
        matches = []
        for i in range(len(track_ids)):
            for j in range(len(detections)):
                if iou_matrix[i, j] > self.iou_threshold:
                    matches.append((i, j, iou_matrix[i, j]))

        matches.sort(key=lambda x: x[2], reverse=True)

        matched_detection_indices = set()
        for i, j, iou in matches:
            track_id = track_ids[i]
            if track_id not in matched_tracks and j not in matched_detection_indices:
                matched_tracks[track_id] = detections[j]
                matched_detection_indices.add(j)

        # Get unmatched detections
        unmatched_detections = [
            det for idx, det in enumerate(detections)
            if idx not in matched_detection_indices
        ]

        return matched_tracks, unmatched_detections

    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes

        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)

        Returns:
            IoU value (0-1)
        """
        # Calculate intersection area
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union area
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def _remove_old_tracks(self):
        """Remove tracks that have been lost for too long"""
        tracks_to_remove = []

        for track_id, track in self.tracks.items():
            if track.frames_since_update > self.max_age:
                track.mark_finished()
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            logger.debug(f"Removing track {track_id} (age: {self.tracks[track_id].frames_since_update})")
            del self.tracks[track_id]

    def get_active_tracks(self) -> List[Track]:
        """
        Get currently active tracks (with minimum hits)

        Returns:
            List of active Track objects
        """
        active_tracks = []

        for track in self.tracks.values():
            if track.state == 'active' and track.total_frames >= self.min_hits:
                active_tracks.append(track)

        return active_tracks

    def get_all_tracks(self) -> List[Track]:
        """
        Get all tracks

        Returns:
            List of all Track objects
        """
        return list(self.tracks.values())

    def get_track(self, track_id: int) -> Optional[Track]:
        """
        Get track by ID

        Args:
            track_id: Track ID

        Returns:
            Track object or None
        """
        return self.tracks.get(track_id)

    def get_track_count(self) -> int:
        """Get number of active tracks"""
        return len(self.tracks)

    def get_stats(self) -> dict:
        """
        Get tracking statistics

        Returns:
            Dictionary with tracking stats
        """
        stats = {
            'total_tracks': len(self.tracks),
            'active_tracks': len(self.get_active_tracks()),
            'frame_count': self.frame_count,
            'next_track_id': self.next_track_id
        }

        # Count by class
        class_counts = defaultdict(int)
        for track in self.get_active_tracks():
            class_counts[track.class_name] += 1

        stats['tracks_by_class'] = dict(class_counts)

        return stats
