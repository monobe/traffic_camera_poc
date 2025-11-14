#!/usr/bin/env python3
"""
Camera connection test tool

Test RTSP connection and display live stream from TP-Link Tapo C320WS
"""

import argparse
import logging
import sys
import time

import cv2
import yaml

from capture.stream import CameraStream


def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_connection(rtsp_url: str, timeout: int = 10) -> bool:
    """
    Test basic camera connection

    Args:
        rtsp_url: RTSP URL
        timeout: Connection timeout in seconds

    Returns:
        True if connection successful
    """
    logger = logging.getLogger(__name__)
    logger.info("Testing camera connection...")

    stream = CameraStream(rtsp_url=rtsp_url)

    try:
        if not stream.connect():
            logger.error("Failed to connect to camera")
            return False

        logger.info("✓ Connection successful")

        # Get stream properties
        props = stream.get_properties()
        logger.info(f"✓ Stream resolution: {props['width']}x{props['height']}")
        logger.info(f"✓ Stream FPS: {props.get('fps', 'N/A')}")

        # Try to read a few frames
        logger.info("Reading test frames...")
        for i in range(5):
            frame = stream.get_frame()
            if frame is None:
                logger.error(f"Failed to read frame {i+1}")
                return False
            logger.info(f"✓ Frame {i+1} read successfully ({frame.shape})")
            time.sleep(0.1)

        logger.info("✓ All tests passed!")
        return True

    except Exception as e:
        logger.error(f"Error during connection test: {e}")
        return False
    finally:
        stream.release()


def display_stream(
    rtsp_url: str,
    fps: int = 15,
    window_name: str = "Camera Stream"
):
    """
    Display live camera stream

    Args:
        rtsp_url: RTSP URL
        fps: Target FPS
        window_name: OpenCV window name
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting live stream display...")
    logger.info("Press 'q' to quit, 's' to save snapshot")

    stream = CameraStream(rtsp_url=rtsp_url, fps=fps)

    try:
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_count = 0
        start_time = time.time()

        for frame in stream.frame_generator(auto_reconnect=True):
            frame_count += 1

            # Calculate actual FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                actual_fps = frame_count / elapsed
            else:
                actual_fps = 0

            # Draw FPS counter
            text = f"FPS: {actual_fps:.1f} | Frame: {frame_count}"
            cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Display frame
            cv2.imshow(window_name, frame)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit requested by user")
                break
            elif key == ord('s'):
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                logger.info(f"Snapshot saved: {filename}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during stream display: {e}")
    finally:
        stream.release()
        cv2.destroyAllWindows()
        logger.info(f"Total frames processed: {frame_count}")


def main():
    """Main entry point"""
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description='Test camera connection and display stream'
    )
    parser.add_argument(
        '--rtsp-url',
        type=str,
        help='RTSP URL (e.g., rtsp://admin:password@192.168.1.100:554/stream1)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Test connection only, do not display stream'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=15,
        help='Target FPS for display (default: 15)'
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

    logger.info("=" * 60)
    logger.info("Camera Connection Test")
    logger.info("=" * 60)

    # Test connection
    if not test_connection(rtsp_url):
        logger.error("Camera connection test failed")
        sys.exit(1)

    # Display stream if requested
    if not args.test_only:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Live Stream Display")
        logger.info("=" * 60)
        display_stream(rtsp_url, fps=args.fps)


if __name__ == "__main__":
    main()
