#!/usr/bin/env python3
"""
Camera calibration tool for speed measurement

Interactive tool to calibrate pixel-to-meter conversion for speed estimation.
User selects two points on the road with known real-world distance.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent.parent))
from capture.stream import CameraStream


logger = logging.getLogger(__name__)


class CameraCalibrator:
    """Interactive camera calibration tool"""

    def __init__(self, rtsp_url: str):
        """
        Initialize calibrator

        Args:
            rtsp_url: RTSP URL of the camera
        """
        self.rtsp_url = rtsp_url
        self.stream = CameraStream(rtsp_url=rtsp_url, fps=1)

        self.points: List[Tuple[int, int]] = []
        self.frame: Optional[np.ndarray] = None
        self.real_distance: Optional[float] = None
        self.pixels_per_meter: Optional[float] = None

        self.window_name = "Camera Calibration"

    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback for point selection

        Args:
            event: Mouse event type
            x, y: Mouse coordinates
            flags: Event flags
            param: Additional parameters
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                logger.info(f"Point {len(self.points)} selected: ({x}, {y})")

                # Redraw frame with points
                self.draw_points()

    def draw_points(self):
        """Draw selected points on the frame"""
        if self.frame is None:
            return

        display_frame = self.frame.copy()

        # Draw points
        for i, point in enumerate(self.points):
            cv2.circle(display_frame, point, 8, (0, 255, 0), -1)
            cv2.putText(
                display_frame,
                f"P{i+1}",
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # Draw line between points
        if len(self.points) == 2:
            cv2.line(display_frame, self.points[0], self.points[1], (0, 255, 0), 2)

            # Calculate pixel distance
            pixel_distance = self.calculate_pixel_distance()
            mid_point = (
                (self.points[0][0] + self.points[1][0]) // 2,
                (self.points[0][1] + self.points[1][1]) // 2
            )
            cv2.putText(
                display_frame,
                f"{pixel_distance:.1f} pixels",
                mid_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # Draw instructions
        instructions = [
            "Click two points on the road with known distance",
            f"Points selected: {len(self.points)}/2",
            "Press 'r' to reset, 'q' to quit"
        ]

        y_offset = 30
        for instruction in instructions:
            cv2.putText(
                display_frame,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            cv2.putText(
                display_frame,
                instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1
            )
            y_offset += 30

        cv2.imshow(self.window_name, display_frame)

    def calculate_pixel_distance(self) -> float:
        """
        Calculate Euclidean distance between two points in pixels

        Returns:
            Distance in pixels
        """
        if len(self.points) != 2:
            return 0.0

        p1, p2 = self.points
        distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return distance

    def calibrate(self) -> bool:
        """
        Run interactive calibration

        Returns:
            True if calibration successful, False otherwise
        """
        logger.info("Starting calibration...")

        # Connect to camera
        if not self.stream.connect():
            logger.error("Failed to connect to camera")
            return False

        # Get a frame
        logger.info("Capturing calibration frame...")
        self.frame = self.stream.get_frame()

        if self.frame is None:
            logger.error("Failed to capture frame")
            return False

        # Save calibration frame
        cv2.imwrite("calibration_frame.jpg", self.frame)
        logger.info("Calibration frame saved as calibration_frame.jpg")

        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Display frame
        self.draw_points()

        logger.info("Click two points on the road with known distance")
        logger.info("Press 'r' to reset points, 'q' to quit")

        # Wait for user to select points
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                logger.info("Calibration cancelled by user")
                cv2.destroyAllWindows()
                return False
            elif key == ord('r'):
                logger.info("Resetting points...")
                self.points = []
                self.draw_points()
            elif key == 13:  # Enter key
                if len(self.points) == 2:
                    break

            # Auto-proceed when 2 points selected
            if len(self.points) == 2:
                # Wait a bit for user to review
                cv2.waitKey(1000)
                break

        cv2.destroyAllWindows()

        # Get real-world distance from user
        print("\n" + "="*60)
        print("Point selection complete!")
        print("="*60)
        print(f"Point 1: {self.points[0]}")
        print(f"Point 2: {self.points[1]}")
        print(f"Pixel distance: {self.calculate_pixel_distance():.1f} pixels")
        print("\nNow enter the REAL-WORLD distance between these two points.")
        print("Measure the distance on the road (use Google Maps satellite view).")
        print("="*60)

        while True:
            try:
                distance_input = input("Enter distance in meters (e.g., 10.5): ")
                self.real_distance = float(distance_input)
                if self.real_distance <= 0:
                    print("Distance must be positive. Try again.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                logger.info("\nCalibration cancelled by user")
                return False

        # Calculate pixels per meter
        pixel_distance = self.calculate_pixel_distance()
        self.pixels_per_meter = pixel_distance / self.real_distance

        logger.info(f"Calibration complete!")
        logger.info(f"Real distance: {self.real_distance:.2f} meters")
        logger.info(f"Pixel distance: {pixel_distance:.1f} pixels")
        logger.info(f"Pixels per meter: {self.pixels_per_meter:.2f}")

        return True

    def save_calibration(self, output_path: str):
        """
        Save calibration data to JSON file

        Args:
            output_path: Path to output JSON file
        """
        if self.pixels_per_meter is None:
            logger.error("No calibration data to save")
            return

        calibration_data = {
            "point1": list(self.points[0]),
            "point2": list(self.points[1]),
            "real_distance_meters": self.real_distance,
            "pixel_distance": float(self.calculate_pixel_distance()),
            "pixels_per_meter": self.pixels_per_meter,
            "calibration_date": datetime.now().isoformat(),
            "rtsp_url_masked": self.stream._mask_credentials(self.rtsp_url),
            "notes": "Calibrated using interactive tool. Points marked on calibration_frame.jpg"
        }

        with open(output_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        logger.info(f"Calibration saved to: {output_path}")
        print(f"\n✓ Calibration saved to: {output_path}")

    def cleanup(self):
        """Release resources"""
        self.stream.release()


def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Camera calibration tool for speed measurement'
    )
    parser.add_argument(
        '--rtsp-url',
        type=str,
        help='RTSP URL of the camera'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='calibration.json',
        help='Output calibration file (default: calibration.json)'
    )

    args = parser.parse_args()

    # Get RTSP URL
    rtsp_url = args.rtsp_url

    if rtsp_url is None:
        # Try to load from config
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
                rtsp_url = config.get('camera', {}).get('rtsp_url')
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    if rtsp_url is None:
        logger.error("No RTSP URL provided. Use --rtsp-url or configure in config.yaml")
        sys.exit(1)

    print("="*60)
    print("Camera Calibration Tool")
    print("="*60)
    print("\nThis tool will help you calibrate the camera for speed measurement.")
    print("\nSteps:")
    print("1. A frame from the camera will be displayed")
    print("2. Click two points on the road with known distance")
    print("   (e.g., two lane markers, road signs, etc.)")
    print("3. Measure the real-world distance between these points")
    print("   (use Google Maps satellite view for accuracy)")
    print("4. Enter the distance in meters")
    print("\nTips:")
    print("- Choose points that are clearly visible")
    print("- Points should be roughly parallel to vehicle movement")
    print("- Longer distance = better accuracy")
    print("="*60)
    input("\nPress Enter to continue...")

    # Run calibration
    calibrator = CameraCalibrator(rtsp_url=rtsp_url)

    try:
        if calibrator.calibrate():
            calibrator.save_calibration(args.output)
            print("\n" + "="*60)
            print("Calibration Complete!")
            print("="*60)
            print(f"\nCalibration file: {args.output}")
            print(f"Calibration frame: calibration_frame.jpg")
            print("\nYou can now use this calibration for speed measurement.")
            print("="*60)
        else:
            logger.error("Calibration failed")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nCalibration interrupted by user")
        sys.exit(1)
    finally:
        calibrator.cleanup()


if __name__ == "__main__":
    main()
