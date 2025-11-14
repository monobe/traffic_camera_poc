#!/usr/bin/env python3
"""
Detection module test tool

Test YOLOv8 object detection on camera stream
"""

import argparse
import logging
import sys
import time

import cv2
import yaml

from capture.stream import CameraStream
from detection.detector import YOLODetector


def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_detection(
    rtsp_url: str,
    model: str = "yolov8n.pt",
    confidence: float = 0.5,
    device: str = "cpu",
    fps: int = 15,
    display: bool = True
):
    """
    Test object detection on camera stream

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
    logger.info("Object Detection Test")
    logger.info("="*60)

    # Initialize detector
    logger.info("Initializing YOLO detector...")
    detector = YOLODetector(
        model_path=model,
        confidence=confidence,
        device=device
    )

    # Load model
    if not detector.load_model():
        logger.error("Failed to load YOLO model")
        return False

    # Initialize camera stream
    logger.info("Connecting to camera...")
    stream = CameraStream(rtsp_url=rtsp_url, fps=fps)

    if not stream.connect():
        logger.error("Failed to connect to camera")
        return False

    logger.info("✓ Setup complete")
    logger.info("")
    logger.info("Starting detection...")
    if display:
        logger.info("Press 'q' to quit, 's' to save snapshot")
    else:
        logger.info("Press Ctrl+C to stop")

    window_name = "Object Detection"
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    detection_count = 0
    start_time = time.time()

    try:
        for frame in stream.frame_generator(auto_reconnect=True):
            frame_count += 1

            # Run detection
            if display:
                annotated_frame, detections = detector.detect_and_visualize(frame)
            else:
                detections = detector.detect(frame)
                annotated_frame = frame

            detection_count += len(detections)

            # Calculate stats
            elapsed = time.time() - start_time
            if elapsed > 0:
                actual_fps = frame_count / elapsed
                avg_detections = detection_count / frame_count
            else:
                actual_fps = 0
                avg_detections = 0

            # Print detections
            if detections:
                det_summary = {}
                for det in detections:
                    det_summary[det.class_name] = det_summary.get(det.class_name, 0) + 1

                det_str = ", ".join([f"{k}: {v}" for k, v in det_summary.items()])
                logger.info(f"Frame {frame_count}: {det_str}")

            # Display frame
            if display:
                # Add stats overlay
                stats_text = [
                    f"FPS: {actual_fps:.1f}",
                    f"Frame: {frame_count}",
                    f"Detections: {len(detections)}",
                    f"Avg: {avg_detections:.1f}/frame"
                ]

                y_offset = 30
                for text in stats_text:
                    cv2.putText(
                        annotated_frame,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    cv2.putText(
                        annotated_frame,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        1
                    )
                    y_offset += 30

                cv2.imshow(window_name, annotated_frame)

                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    break
                elif key == ord('s'):
                    filename = f"detection_snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    logger.info(f"Snapshot saved: {filename}")

            # Show periodic stats (no display mode)
            if not display and frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames, "
                           f"{actual_fps:.1f} FPS, "
                           f"{avg_detections:.1f} detections/frame")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during detection: {e}", exc_info=True)
    finally:
        stream.release()
        if display:
            cv2.destroyAllWindows()

        # Final stats
        elapsed = time.time() - start_time
        logger.info("")
        logger.info("="*60)
        logger.info("Detection Test Summary")
        logger.info("="*60)
        logger.info(f"Total frames: {frame_count}")
        logger.info(f"Total detections: {detection_count}")
        logger.info(f"Average FPS: {frame_count/elapsed:.1f}")
        logger.info(f"Average detections per frame: {detection_count/frame_count:.1f}")
        logger.info("="*60)

    return True


def main():
    """Main entry point"""
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description='Test object detection on camera stream'
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
    success = test_detection(
        rtsp_url=rtsp_url,
        model=args.model,
        confidence=args.confidence,
        device=args.device,
        fps=args.fps,
        display=not args.no_display
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
