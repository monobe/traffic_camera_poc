#!/usr/bin/env python3
"""
Tracking module test tool

Test object detection + tracking on camera stream
"""

import argparse
import logging
import sys
import time

import cv2
import yaml

from capture.stream import CameraStream
from detection.detector import YOLODetector
from tracking.tracker import ObjectTracker


def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_tracking(
    rtsp_url: str,
    model: str = "yolov8n.pt",
    confidence: float = 0.5,
    device: str = "cpu",
    fps: int = 15,
    display: bool = True
):
    """
    Test object tracking on camera stream

    Args:
        rtsp_url: Camera RTSP URL
        model: YOLO model path
        confidence: Detection confidence threshold
        device: Device to run on
        fps: Target FPS
        display: Display annotated stream
    """
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info("Object Detection + Tracking Test")
    logger.info("="*60)

    # Initialize detector
    logger.info("Initializing YOLO detector...")
    detector = YOLODetector(
        model_path=model,
        confidence=confidence,
        device=device
    )

    if not detector.load_model():
        logger.error("Failed to load YOLO model")
        return False

    # Initialize tracker
    logger.info("Initializing object tracker...")
    tracker = ObjectTracker(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3
    )

    # Initialize camera stream
    logger.info("Connecting to camera...")
    stream = CameraStream(rtsp_url=rtsp_url, fps=fps)

    if not stream.connect():
        logger.error("Failed to connect to camera")
        return False

    logger.info("✓ Setup complete")
    logger.info("")
    logger.info("Starting tracking...")
    if display:
        logger.info("Press 'q' to quit, 's' to save snapshot")
    else:
        logger.info("Press Ctrl+C to stop")

    window_name = "Object Tracking"
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Define colors for tracks
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 165, 255),  # Orange
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    frame_count = 0
    start_time = time.time()

    try:
        for frame in stream.frame_generator(auto_reconnect=True):
            frame_count += 1

            # Run detection
            detections = detector.detect(frame)

            # Update tracker
            tracks = tracker.update(detections)

            # Calculate stats
            elapsed = time.time() - start_time
            if elapsed > 0:
                actual_fps = frame_count / elapsed
            else:
                actual_fps = 0

            # Log tracking info
            if tracks:
                track_summary = {}
                for track in tracks:
                    track_summary[track.class_name] = track_summary.get(track.class_name, 0) + 1

                track_str = ", ".join([f"{k}: {v}" for k, v in track_summary.items()])
                logger.info(f"Frame {frame_count}: Tracking {len(tracks)} objects ({track_str})")

            # Display frame
            if display:
                annotated_frame = frame.copy()

                # Draw tracks
                for track in tracks:
                    # Get color for this track
                    color = colors[track.track_id % len(colors)]

                    # Draw trajectory
                    if len(track.trajectory) > 1:
                        points = np.array(track.trajectory, np.int32)
                        cv2.polylines(annotated_frame, [points], False, color, 2)

                    # Draw current bounding box
                    bbox = track.get_last_bbox()
                    if bbox:
                        cv2.rectangle(
                            annotated_frame,
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            color,
                            2
                        )

                        # Draw track info
                        label = f"ID:{track.track_id} {track.class_name} [{track.total_frames}]"

                        # Draw label background
                        (label_width, label_height), _ = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            1
                        )

                        cv2.rectangle(
                            annotated_frame,
                            (bbox[0], bbox[1] - label_height - 10),
                            (bbox[0] + label_width, bbox[1]),
                            color,
                            -1
                        )

                        # Draw label text
                        cv2.putText(
                            annotated_frame,
                            label,
                            (bbox[0], bbox[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            1
                        )

                # Add stats overlay
                stats = tracker.get_stats()
                stats_text = [
                    f"FPS: {actual_fps:.1f}",
                    f"Frame: {frame_count}",
                    f"Detections: {len(detections)}",
                    f"Active Tracks: {stats['active_tracks']}",
                    f"Total Tracks: {stats['next_track_id'] - 1}"
                ]

                y_offset = 30
                for text in stats_text:
                    cv2.putText(
                        annotated_frame,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
                    cv2.putText(
                        annotated_frame,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        1
                    )
                    y_offset += 25

                cv2.imshow(window_name, annotated_frame)

                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    break
                elif key == ord('s'):
                    filename = f"tracking_snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    logger.info(f"Snapshot saved: {filename}")

            # Show periodic stats (no display mode)
            if not display and frame_count % 30 == 0:
                stats = tracker.get_stats()
                logger.info(f"Processed {frame_count} frames, "
                           f"{actual_fps:.1f} FPS, "
                           f"{stats['active_tracks']} active tracks")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during tracking: {e}", exc_info=True)
    finally:
        stream.release()
        if display:
            cv2.destroyAllWindows()

        # Final stats
        stats = tracker.get_stats()
        elapsed = time.time() - start_time
        logger.info("")
        logger.info("="*60)
        logger.info("Tracking Test Summary")
        logger.info("="*60)
        logger.info(f"Total frames: {frame_count}")
        logger.info(f"Average FPS: {frame_count/elapsed:.1f}")
        logger.info(f"Total tracks created: {stats['next_track_id'] - 1}")
        logger.info(f"Final active tracks: {stats['active_tracks']}")
        if stats.get('tracks_by_class'):
            logger.info(f"Tracks by class: {stats['tracks_by_class']}")
        logger.info("="*60)

    return True


def main():
    """Main entry point"""
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description='Test object detection and tracking on camera stream'
    )
    parser.add_argument(
        '--rtsp-url',
        type=str,
        help='RTSP URL'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='YOLO model (default: yolov8n.pt)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device (cpu or cuda, default: cpu)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=15,
        help='Target FPS (default: 15)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display video (headless mode)'
    )

    args = parser.parse_args()

    # Get RTSP URL
    rtsp_url = args.rtsp_url

    if rtsp_url is None:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
                rtsp_url = config.get('camera', {}).get('rtsp_url')
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    if rtsp_url is None:
        logger.error("No RTSP URL provided")
        sys.exit(1)

    # Run test
    success = test_tracking(
        rtsp_url=rtsp_url,
        model=args.model,
        confidence=args.confidence,
        device=args.device,
        fps=args.fps,
        display=not args.no_display
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    import numpy as np
    main()
