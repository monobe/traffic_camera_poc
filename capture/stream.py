"""
Camera stream capture module

Handles RTSP stream capture from IP cameras (TP-Link Tapo C320WS)
"""

import logging
import time
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class CameraStream:
    """RTSP stream handler for IP cameras"""

    def __init__(
        self,
        rtsp_url: str,
        fps: int = 15,
        resolution: Optional[Tuple[int, int]] = None,
        reconnect_timeout: int = 30,
        buffer_size: int = 1
    ):
        """
        Initialize camera stream

        Args:
            rtsp_url: RTSP URL (e.g., rtsp://user:pass@192.168.1.100:554/stream1)
            fps: Target frames per second
            resolution: Target resolution (width, height), None for original
            reconnect_timeout: Timeout before reconnection attempt (seconds)
            buffer_size: OpenCV buffer size (1 = minimal latency)
        """
        self.rtsp_url = rtsp_url
        self.fps = fps
        self.resolution = resolution
        self.reconnect_timeout = reconnect_timeout
        self.buffer_size = buffer_size

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.frame_count = 0
        self.last_reconnect_time = 0

    def connect(self) -> bool:
        """
        Connect to camera

        Returns:
            True if connection successful, False otherwise
        """
        logger.info(f"Connecting to camera: {self._mask_credentials(self.rtsp_url)}")

        try:
            # Create VideoCapture object
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            # Set buffer size (reduce latency)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

            # Check if opened successfully
            if not self.cap.isOpened():
                logger.error("Failed to open camera stream")
                return False

            # Try to read a test frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("Failed to read test frame from camera")
                self.cap.release()
                return False

            self.is_connected = True
            logger.info(f"Successfully connected to camera")
            logger.info(f"Stream resolution: {frame.shape[1]}x{frame.shape[0]}")

            return True

        except Exception as e:
            logger.error(f"Error connecting to camera: {e}")
            return False

    def reconnect(self) -> bool:
        """
        Reconnect to camera

        Returns:
            True if reconnection successful, False otherwise
        """
        current_time = time.time()

        # Check if we should wait before reconnecting
        if current_time - self.last_reconnect_time < self.reconnect_timeout:
            return False

        logger.warning("Attempting to reconnect to camera...")
        self.last_reconnect_time = current_time

        # Release existing connection
        if self.cap is not None:
            self.cap.release()

        # Try to reconnect
        return self.connect()

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get single frame from camera

        Returns:
            Frame as numpy array, or None if failed
        """
        if not self.is_connected:
            logger.warning("Camera not connected")
            return None

        try:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                logger.warning("Failed to read frame from camera")
                self.is_connected = False
                return None

            # Resize if needed
            if self.resolution is not None:
                frame = cv2.resize(frame, self.resolution)

            self.frame_count += 1
            return frame

        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            self.is_connected = False
            return None

    def frame_generator(self, auto_reconnect: bool = True) -> Iterator[np.ndarray]:
        """
        Yield frames continuously

        Args:
            auto_reconnect: Automatically reconnect on connection loss

        Yields:
            Frames as numpy arrays
        """
        # Initial connection
        if not self.is_connected:
            if not self.connect():
                logger.error("Failed to connect to camera")
                return

        # Calculate frame delay for target FPS
        frame_delay = 1.0 / self.fps

        while True:
            start_time = time.time()

            # Get frame
            frame = self.get_frame()

            if frame is None:
                # Try to reconnect if enabled
                if auto_reconnect and self.reconnect():
                    logger.info("Reconnected successfully")
                    continue
                else:
                    logger.error("Failed to get frame, stopping...")
                    break

            yield frame

            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_properties(self) -> dict:
        """
        Get camera stream properties

        Returns:
            Dictionary of stream properties
        """
        if self.cap is None or not self.is_connected:
            return {}

        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'codec': int(self.cap.get(cv2.CAP_PROP_FOURCC)),
            'frame_count': self.frame_count,
        }

    def release(self):
        """Release camera connection"""
        if self.cap is not None:
            logger.info("Releasing camera connection")
            self.cap.release()
            self.is_connected = False

    def _mask_credentials(self, url: str) -> str:
        """
        Mask credentials in RTSP URL for logging

        Args:
            url: RTSP URL with credentials

        Returns:
            URL with masked credentials
        """
        if '@' in url:
            # rtsp://user:pass@host:port/path -> rtsp://***:***@host:port/path
            protocol, rest = url.split('://', 1)
            if '@' in rest:
                credentials, host_part = rest.split('@', 1)
                return f"{protocol}://***:***@{host_part}"
        return url

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
