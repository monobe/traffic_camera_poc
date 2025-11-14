#!/usr/bin/env python3
"""
Test vehicle classification feature

Demonstrates the detailed vehicle type classification capability
"""

import cv2
import yaml
from pathlib import Path

from capture.stream import CameraStream
from detection.detector import YOLODetector
from detection.classifier import VehicleClassifier


def main():
    """Test classification on camera feed"""

    # Load config
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("Error: config.yaml not found")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize camera
    camera_config = config.get('camera', {})
    camera = CameraStream(
        rtsp_url=camera_config.get('rtsp_url'),
        fps=camera_config.get('fps', 10)
    )

    # Initialize detector with classification enabled
    detection_config = config.get('detection', {})
    detector = YOLODetector(
        model_path=detection_config.get('model', 'yolov8n.pt'),
        confidence=detection_config.get('confidence', 0.5),
        device=detection_config.get('device', 'cpu'),
        enable_classification=True  # Enable detailed classification
    )

    print("=" * 60)
    print("Vehicle Classification Test")
    print("=" * 60)
    print("\nConnecting to camera...")

    if not camera.connect():
        print("Failed to connect to camera")
        return

    print("Loading YOLO model...")
    if not detector.load_model():
        print("Failed to load YOLO model")
        return

    print("\nProcessing frames...")
    print("Press 'q' to quit\n")

    frame_count = 0
    detection_count = 0
    classification_stats = {}

    try:
        for frame in camera.frame_generator(auto_reconnect=True):
            frame_count += 1

            # Run detection
            detections = detector.detect(frame)
            detection_count += len(detections)

            # Count classifications
            for det in detections:
                if det.subtype:
                    classification_stats[det.subtype] = classification_stats.get(det.subtype, 0) + 1

                    # Print detailed classification
                    display_name = detector.classifier.get_display_name(det.subtype, language='ja')
                    print(f"Detected: {display_name} (base: {det.class_name}, "
                          f"confidence: {det.confidence:.2f}, "
                          f"subtype_conf: {det.subtype_confidence:.2f})")

            # Visualize
            annotated_frame, _ = detector.detect_and_visualize(frame)

            # Add stats overlay
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count} | Detections: {detection_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Display
            cv2.imshow('Vehicle Classification Test', annotated_frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Limit frames for testing
            if frame_count >= 100:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        camera.release()
        cv2.destroyAllWindows()

    # Print statistics
    print("\n" + "=" * 60)
    print("Classification Statistics")
    print("=" * 60)
    print(f"Total frames: {frame_count}")
    print(f"Total detections: {detection_count}")
    print("\nVehicle subtypes detected:")

    for subtype, count in sorted(classification_stats.items(), key=lambda x: x[1], reverse=True):
        display_name = detector.classifier.get_display_name(subtype, language='ja')
        print(f"  {display_name} ({subtype}): {count}")


if __name__ == "__main__":
    main()
