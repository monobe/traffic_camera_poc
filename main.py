#!/usr/bin/env python3
"""
Traffic Camera PoC - Main Entry Point

AI-powered traffic monitoring system for residential areas.
Monitors vehicle speed and traffic volume, generates reports for authorities.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from capture.stream import CameraStream, ThreadedCameraStream
from detection.detector import YOLODetector
from tracking.tracker import ObjectTracker
from speed_estimation.estimator import SpeedEstimator
from storage.database import StorageManager


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))

    handlers = []

    # File handler
    log_file = log_config.get('file', './logs/traffic_monitor.log')
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    handlers.append(file_handler)

    # Console handler
    if log_config.get('console', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        handlers.append(console_handler)

    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in configuration file: {e}")
        sys.exit(1)


class TrafficMonitor:
    """Main traffic monitoring system"""

    def __init__(self, config: dict):
        """
        Initialize traffic monitor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.camera = None
        self.detector = None
        self.tracker = None
        self.speed_estimator = None
        self.storage = None

        self.frame_count = 0
        self.detection_count = 0
        self.speed_estimate_count = 0

        self._init_components()

    def _init_components(self):
        """Initialize all system components"""
        self.logger.info("Initializing system components...")

        # 1. Camera
        camera_config = self.config.get('camera', {})
        use_threading = self.config.get('performance', {}).get('threaded_capture', True)
        
        stream_class = ThreadedCameraStream if use_threading else CameraStream
        self.logger.info(f"Using {'threaded' if use_threading else 'standard'} camera stream")
        
        self.camera = stream_class(
            rtsp_url=camera_config.get('rtsp_url'),
            fps=camera_config.get('fps', 15),
            resolution=tuple(camera_config.get('resolution')) if camera_config.get('resolution') else None,
            reconnect_timeout=camera_config.get('reconnect_timeout', 30),
            buffer_size=camera_config.get('buffer_size', 1)
        )

        # 2. Detector
        detection_config = self.config.get('detection', {})
        self.detector = YOLODetector(
            model_path=detection_config.get('model', 'yolov8n.pt'),
            confidence=detection_config.get('confidence', 0.5),
            device=detection_config.get('device', 'cpu'),
            classes=detection_config.get('classes'),
            imgsz=detection_config.get('imgsz', 640),
            enable_classification=detection_config.get('enable_classification', True)
        )

        # 3. Tracker
        tracking_config = self.config.get('tracking', {})
        self.tracker = ObjectTracker(
            max_age=tracking_config.get('max_age', 30),
            min_hits=tracking_config.get('min_hits', 3),
            iou_threshold=tracking_config.get('iou_threshold', 0.3)
        )

        # 4. Speed Estimator
        speed_config = self.config.get('speed_estimation', {})
        self.speed_estimator = SpeedEstimator(
            calibration_file=speed_config.get('calibration_file', 'calibration.json'),
            fps=camera_config.get('fps', 15),
            min_track_length=speed_config.get('min_track_length', 10),
            speed_limit_kmh=speed_config.get('speed_limit_kmh', 30.0),
            direction_threshold=speed_config.get('direction_threshold', 50)
        )

        # 5. Storage
        storage_config = self.config.get('storage', {})
        self.storage = StorageManager(
            db_path=storage_config.get('database', './data/traffic.db'),
            csv_dir=storage_config.get('csv_dir', './data/csv'),
            retention_days=storage_config.get('retention_days', 90)
        )

        self.logger.info("✓ All components initialized")

    def run(self):
        """Main processing loop"""
        self.logger.info("Connecting to camera...")

        if not self.camera.connect():
            self.logger.error("Failed to connect to camera")
            return False

        self.logger.info("Loading YOLO model...")
        if not self.detector.load_model():
            self.logger.error("Failed to load YOLO model")
            return False

        self.logger.info("=" * 60)
        self.logger.info("Traffic monitoring started")
        self.logger.info("Press Ctrl+C to stop")
        self.logger.info("=" * 60)

        start_time = time.time()
        last_stats_time = time.time()
        stats_interval = 60  # Print stats every 60 seconds

        # Track finished tracks to avoid duplicate speed estimation
        estimated_tracks = set()

        try:
            frame_skip = self.config.get('performance', {}).get('frame_skip', 1)
            
            for frame in self.camera.frame_generator(auto_reconnect=True):
                self.frame_count += 1
                
                # Frame skipping
                if self.frame_count % frame_skip != 0:
                    continue

                # Run detection
                detections = self.detector.detect(frame)
                self.detection_count += len(detections)

                # Update tracker
                tracks = self.tracker.update(detections)

                # Process tracks for speed estimation
                for track in tracks:
                    # Only estimate speed once when track is long enough
                    if track.track_id not in estimated_tracks and track.total_frames >= self.speed_estimator.min_track_length:
                        speed_estimate = self.speed_estimator.estimate_speed(track)

                        if speed_estimate and self.speed_estimator.is_speed_valid(speed_estimate.speed_kmh):
                            # Save to database and CSV
                            self.storage.save_detection(speed_estimate)
                            speed_limit = self.config.get('speed_estimation', {}).get('speed_limit_kmh', 30.0)
                            self.storage.save_csv(speed_estimate, speed_limit)

                            self.speed_estimate_count += 1
                            estimated_tracks.add(track.track_id)

                            # Log speed estimate
                            report = self.speed_estimator.format_speed_report(speed_estimate)
                            self.logger.info(report)

                # Print periodic stats
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    elapsed = current_time - start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0

                    tracker_stats = self.tracker.get_stats()

                    self.logger.info("-" * 60)
                    self.logger.info(f"Stats: {self.frame_count} frames, {fps:.1f} FPS, "
                                   f"{self.detection_count} detections, "
                                   f"{self.speed_estimate_count} speed estimates, "
                                   f"{tracker_stats['active_tracks']} active tracks")
                    self.logger.info("-" * 60)

                    last_stats_time = current_time

                    # Calculate hourly/daily stats periodically
                    now = datetime.now()
                    if now.minute == 0:  # Every hour
                        self.storage.calculate_hourly_stats(now)
                        self.storage.calculate_daily_stats(now)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during processing: {e}", exc_info=True)
            return False
        finally:
            self.cleanup()

        return True

    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up...")

        if self.camera:
            self.camera.release()

        if self.storage:
            self.storage.close()

        elapsed = time.time() - self.frame_count if self.frame_count > 0 else 0
        self.logger.info("=" * 60)
        self.logger.info("Final Statistics")
        self.logger.info("=" * 60)
        self.logger.info(f"Total frames processed: {self.frame_count}")
        self.logger.info(f"Total detections: {self.detection_count}")
        self.logger.info(f"Speed estimates saved: {self.speed_estimate_count}")
        self.logger.info("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Traffic Camera PoC - AI-powered traffic monitoring'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Run calibration tool'
    )
    parser.add_argument(
        '--test-camera',
        action='store_true',
        help='Test camera connection'
    )
    parser.add_argument(
        '--test-detection',
        action='store_true',
        help='Test detection'
    )
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Start dashboard only'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Traffic Camera PoC")
    logger.info("=" * 60)

    # Calibration mode
    if args.calibrate:
        logger.info("Starting calibration mode...")
        from calibration.calibrate import main as calibrate_main
        calibrate_main()
        return

    # Test camera mode
    if args.test_camera:
        logger.info("Testing camera connection...")
        from test_camera import test_connection
        rtsp_url = config.get('camera', {}).get('rtsp_url')
        if test_connection(rtsp_url):
            logger.info("✓ Camera test passed")
        else:
            logger.error("✗ Camera test failed")
        return

    # Test detection mode
    if args.test_detection:
        logger.info("Testing detection...")
        # Run test_detection.py logic here
        logger.info("Use: python test_detection.py --config config.yaml")
        return

    # Dashboard mode
    if args.dashboard:
        logger.info("Starting dashboard...")
        from dashboard.app import main as dashboard_main
        dashboard_main()
        return

    # Normal operation
    logger.info("Starting traffic monitoring system...")
    logger.info(f"Configuration: {args.config}")

    try:
        monitor = TrafficMonitor(config)
        success = monitor.run()

        if success:
            logger.info("✓ Traffic monitoring completed successfully")
        else:
            logger.error("✗ Traffic monitoring failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
