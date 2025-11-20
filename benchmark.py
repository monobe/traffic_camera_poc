#!/usr/bin/env python3
"""
Benchmark script for Traffic Camera PoC
"""

import time
import argparse
import logging
import cv2
import yaml
import sys
from pathlib import Path

from capture.stream import CameraStream
from detection.detector import YOLODetector
from tracking.tracker import ObjectTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def benchmark(video_path, config_path, device_override=None, model_override=None, max_frames=300):
    """Run benchmark on a video file"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    detection_config = config.get('detection', {})
    
    # Override for benchmark
    if model_override:
        model_path = model_override
    else:
        model_path = detection_config.get('model', 'yolov8n.pt')
        
    confidence = detection_config.get('confidence', 0.5)
    
    # Allow overriding device
    if device_override:
        device = device_override
    else:
        device = detection_config.get('device', 'cpu')
        
    imgsz = detection_config.get('imgsz', 640)
    
    enable_classification = detection_config.get('enable_classification', True)
    
    logger.info(f"Benchmarking with:")
    logger.info(f"  Video: {video_path}")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Image Size: {imgsz}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Classification: {enable_classification}")
    
    # Initialize detector
    detector = YOLODetector(
        model_path=model_path,
        confidence=confidence,
        device=device,
        imgsz=imgsz,
        enable_classification=enable_classification
    )
    
    if not detector.load_model():
        logger.error("Failed to load model")
        return

    # Initialize tracker
    tracking_config = config.get('tracking', {})
    tracker = ObjectTracker(
        max_age=tracking_config.get('max_age', 30),
        min_hits=tracking_config.get('min_hits', 3),
        iou_threshold=tracking_config.get('iou_threshold', 0.3)
    )
        
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
        
    frame_count = 0
    start_time = time.time()
    
    det_times = []
    track_times = []
    
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Detection
            t0 = time.time()
            detections = detector.detect(frame)
            t1 = time.time()
            det_times.append((t1 - t0) * 1000)
            
            # Tracking
            t2 = time.time()
            tracker.update(detections)
            t3 = time.time()
            track_times.append((t3 - t2) * 1000)
            
            if frame_count % 50 == 0:
                logger.info(f"Processed {frame_count} frames...")
                
    finally:
        cap.release()
        
    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time
    avg_det = sum(det_times) / len(det_times) if det_times else 0
    avg_track = sum(track_times) / len(track_times) if track_times else 0
    
    logger.info("=" * 40)
    logger.info("Benchmark Results")
    logger.info("=" * 40)
    logger.info(f"Total Frames: {frame_count}")
    logger.info(f"Total Time: {total_time:.2f}s")
    logger.info(f"Average FPS: {fps:.2f}")
    logger.info(f"Avg Detection Time: {avg_det:.2f}ms")
    logger.info(f"Avg Tracking Time: {avg_track:.2f}ms")
    logger.info(f"Total Latency: {avg_det + avg_track:.2f}ms")
    logger.info("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help='Path to test video')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--device', help='Override device (e.g., cpu, mps)')
    parser.add_argument('--model', help='Override model path')
    args = parser.parse_args()
    
    benchmark(args.video_path, args.config, args.device, args.model)
