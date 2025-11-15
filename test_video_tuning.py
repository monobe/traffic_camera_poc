"""
Video-based Detection Tuning Script

Test and tune detection parameters using recorded video files
"""

import argparse
import cv2
import logging
import sys
import time
from pathlib import Path

import yaml
from detection.detector import YOLODetector
from tracking.tracker import ObjectTracker
from speed_estimation.estimator import SpeedEstimator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Tune detection parameters using video')
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--confidence', type=float, help='Override detection confidence')
    parser.add_argument('--iou-threshold', type=float, help='Override IoU threshold')
    parser.add_argument('--min-track-length', type=int, help='Override min track length')
    parser.add_argument('--output', type=str, help='Save output video with annotations')
    parser.add_argument('--display', action='store_true', help='Display video in window')
    parser.add_argument('--fps', type=int, default=20, help='FPS for speed estimation')
    parser.add_argument('--stats', action='store_true', help='Show detailed statistics')

    args = parser.parse_args()

    # Check video file exists
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override parameters if specified
    if args.confidence is not None:
        config['detection']['confidence'] = args.confidence
        logger.info(f"Override confidence: {args.confidence}")

    if args.iou_threshold is not None:
        config['tracking']['iou_threshold'] = args.iou_threshold
        logger.info(f"Override IoU threshold: {args.iou_threshold}")

    if args.min_track_length is not None:
        config['speed_estimation']['min_track_length'] = args.min_track_length
        logger.info(f"Override min track length: {args.min_track_length}")

    # Initialize detector
    logger.info("Initializing detector...")
    detector = YOLODetector(
        model_path=config['detection']['model'],
        confidence=config['detection']['confidence'],
        device=config['detection']['device'],
        classes=config['detection']['classes']
    )

    # Initialize tracker
    tracker = ObjectTracker(
        max_age=config['tracking']['max_age'],
        min_hits=config['tracking']['min_hits'],
        iou_threshold=config['tracking']['iou_threshold']
    )

    # Initialize speed estimator
    estimator = SpeedEstimator(
        fps=args.fps,
        calibration_file=config['speed_estimation']['calibration_file'],
        min_track_length=config['speed_estimation']['min_track_length'],
        speed_limit_kmh=config['speed_estimation']['speed_limit_kmh']
    )

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        sys.exit(1)

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video: {video_path}")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {video_fps:.2f}")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Duration: {total_frames / video_fps:.1f}s")

    # Setup output video if specified
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, video_fps, (width, height))
        logger.info(f"Output video: {args.output}")

    # Statistics
    stats = {
        'total_detections': 0,
        'total_tracks': 0,
        'speed_estimates': 0,
        'by_class': {}
    }

    frame_count = 0
    start_time = time.time()

    logger.info("\nProcessing video...")
    logger.info("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run detection
            detections = detector.detect(frame)
            stats['total_detections'] += len(detections)

            for det in detections:
                if det.class_name not in stats['by_class']:
                    stats['by_class'][det.class_name] = {
                        'detections': 0,
                        'tracks': 0,
                        'speeds': []
                    }
                stats['by_class'][det.class_name]['detections'] += 1

            # Run tracking
            tracks = tracker.update(detections, frame_count)

            # Estimate speeds
            speed_estimates = []
            for track in tracks:
                estimate = estimator.estimate_speed(track)
                if estimate:
                    speed_estimates.append(estimate)
                    stats['speed_estimates'] += 1
                    if estimate.object_type in stats['by_class']:
                        stats['by_class'][estimate.object_type]['speeds'].append(estimate.speed_kmh)

            # Draw annotations
            annotated_frame = frame.copy()

            # Draw tracks
            for track in tracks:
                if track.frames_since_update > 2:
                    continue

                if track.bbox_history:
                    x1, y1, x2, y2 = track.bbox_history[-1]

                    # Color by class
                    colors = {
                        'car': (0, 255, 0),
                        'motorcycle': (255, 0, 0),
                        'bus': (0, 165, 255),
                        'truck': (0, 255, 255),
                        'bicycle': (255, 255, 0),
                        'person': (255, 0, 255)
                    }
                    color = colors.get(track.class_name, (0, 255, 0))

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    label = f"ID:{track.track_id} {track.class_name}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw speed estimates
            for estimate in speed_estimates:
                # Find track
                track = next((t for t in tracks if t.track_id == estimate.track_id), None)
                if track and track.bbox_history:
                    x1, y1, x2, y2 = track.bbox_history[-1]

                    speed_text = f"{estimate.speed_kmh:.1f} km/h"
                    cv2.putText(annotated_frame, speed_text, (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Show progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) | "
                           f"FPS: {fps:.1f} | Detections: {len(detections)} | "
                           f"Tracks: {len([t for t in tracks if t.frames_since_update <= 2])}")

            # Display if requested
            if args.display:
                cv2.imshow('Detection Tuning', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User quit")
                    break

            # Write to output
            if out:
                out.write(annotated_frame)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")

    finally:
        cap.release()
        if out:
            out.release()
        if args.display:
            cv2.destroyAllWindows()

    # Calculate statistics
    elapsed = time.time() - start_time
    processing_fps = frame_count / elapsed if elapsed > 0 else 0

    # Count unique tracks
    stats['total_tracks'] = len(tracker.tracks) if hasattr(tracker, 'tracks') else tracker.next_id

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Processed: {frame_count} frames in {elapsed:.1f}s")
    logger.info(f"Processing FPS: {processing_fps:.1f}")
    logger.info(f"Total detections: {stats['total_detections']}")
    logger.info(f"Total tracks: {stats['total_tracks']}")
    logger.info(f"Speed estimates: {stats['speed_estimates']}")

    if args.stats and stats['by_class']:
        logger.info("\nBy Class:")
        logger.info("-" * 60)
        for class_name, data in stats['by_class'].items():
            logger.info(f"\n{class_name.upper()}:")
            logger.info(f"  Detections: {data['detections']}")
            if data['speeds']:
                logger.info(f"  Speed estimates: {len(data['speeds'])}")
                logger.info(f"  Avg speed: {sum(data['speeds']) / len(data['speeds']):.1f} km/h")
                logger.info(f"  Max speed: {max(data['speeds']):.1f} km/h")
                logger.info(f"  Min speed: {min(data['speeds']):.1f} km/h")

    logger.info("\n" + "=" * 60)
    logger.info(f"Parameters used:")
    logger.info(f"  Confidence: {config['detection']['confidence']}")
    logger.info(f"  IoU threshold: {config['tracking']['iou_threshold']}")
    logger.info(f"  Min track length: {config['speed_estimation']['min_track_length']}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
