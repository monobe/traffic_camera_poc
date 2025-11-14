#!/usr/bin/env python3
"""
Traffic Monitoring Dashboard

Web-based dashboard for real-time traffic monitoring and historical data analysis
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import cv2
import yaml

# Import application modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from storage.database import StorageManager
from capture.stream import CameraStream
from detection.detector import YOLODetector


logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Traffic Monitoring Dashboard",
    description="Real-time traffic monitoring and analytics",
    version="1.0.0"
)

# Setup templates
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))

# Initialize storage and camera
storage = None
camera_stream = None
detector = None
camera_config = None


def init_storage(db_path: str = "./data/traffic.db", csv_dir: str = "./data/csv"):
    """Initialize storage manager"""
    global storage
    storage = StorageManager(db_path=db_path, csv_dir=csv_dir)
    logger.info("Storage manager initialized")


def init_camera():
    """Initialize camera stream"""
    global camera_stream, detector, camera_config

    try:
        # Load config
        config_path = Path("config.yaml")
        if not config_path.exists():
            logger.warning("config.yaml not found, camera streaming disabled")
            return

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        camera_config = config.get('camera', {})
        rtsp_url = camera_config.get('rtsp_url')

        if not rtsp_url:
            logger.warning("RTSP URL not configured, camera streaming disabled")
            return

        # Initialize camera
        camera_stream = CameraStream(
            rtsp_url=rtsp_url,
            fps=camera_config.get('fps', 10),
            resolution=tuple(camera_config.get('resolution')) if camera_config.get('resolution') else None
        )

        # Initialize detector for visualization
        detection_config = config.get('detection', {})
        detector = YOLODetector(
            model_path=detection_config.get('model', 'yolov8n.pt'),
            confidence=detection_config.get('confidence', 0.5),
            device=detection_config.get('device', 'cpu')
        )

        logger.info("Camera streaming initialized")
    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    init_storage()
    init_camera()
    logger.info("Dashboard started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if storage:
        storage.close()
    if camera_stream:
        camera_stream.release()
    logger.info("Dashboard stopped")


# Routes

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/stats/summary")
async def get_summary_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get summary statistics for date range

    Args:
        start_date: Start date (YYYY-MM-DD), default: 7 days ago
        end_date: End date (YYYY-MM-DD), default: today
    """
    try:
        # Parse dates
        if end_date is None:
            end_dt = datetime.now()
        else:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        if start_date is None:
            start_dt = end_dt - timedelta(days=7)
        else:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')

        # Get data
        summary = storage.get_daily_summary(start_dt, end_dt)

        return {
            "start_date": start_dt.strftime('%Y-%m-%d'),
            "end_date": end_dt.strftime('%Y-%m-%d'),
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Error getting summary stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats/daily")
async def get_daily_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get daily statistics"""
    try:
        # Parse dates
        if end_date is None:
            end_dt = datetime.now()
        else:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        if start_date is None:
            start_dt = end_dt - timedelta(days=7)
        else:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')

        # Get detections
        detections_df = storage.get_detections(start_dt, end_dt)

        if detections_df.empty:
            return {"daily_data": []}

        # Group by date
        import pandas as pd
        detections_df['date'] = pd.to_datetime(detections_df['timestamp']).dt.date

        daily_stats = detections_df.groupby('date').agg({
            'track_id': 'count',
            'speed_kmh': ['mean', 'max', 'min']
        }).reset_index()

        daily_stats.columns = ['date', 'count', 'avg_speed', 'max_speed', 'min_speed']

        # Convert to list of dicts
        daily_data = daily_stats.to_dict('records')
        for item in daily_data:
            item['date'] = str(item['date'])

        return {"daily_data": daily_data}

    except Exception as e:
        logger.error(f"Error getting daily stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats/hourly")
async def get_hourly_stats(date: str):
    """
    Get hourly statistics for a specific date

    Args:
        date: Date (YYYY-MM-DD)
    """
    try:
        target_date = datetime.strptime(date, '%Y-%m-%d')

        # Get detections for the date
        start_dt = target_date.replace(hour=0, minute=0, second=0)
        end_dt = target_date.replace(hour=23, minute=59, second=59)

        detections_df = storage.get_detections(start_dt, end_dt)

        if detections_df.empty:
            return {"hourly_data": []}

        # Group by hour
        import pandas as pd
        detections_df['hour'] = pd.to_datetime(detections_df['timestamp']).dt.hour

        hourly_stats = detections_df.groupby('hour').agg({
            'track_id': 'count',
            'speed_kmh': 'mean'
        }).reset_index()

        hourly_stats.columns = ['hour', 'count', 'avg_speed']

        # Fill missing hours with 0
        all_hours = pd.DataFrame({'hour': range(24)})
        hourly_stats = all_hours.merge(hourly_stats, on='hour', how='left').fillna(0)

        hourly_data = hourly_stats.to_dict('records')

        return {"hourly_data": hourly_data}

    except Exception as e:
        logger.error(f"Error getting hourly stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats/speed_distribution")
async def get_speed_distribution(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    bins: int = 15
):
    """Get speed distribution histogram data"""
    try:
        # Parse dates
        if end_date is None:
            end_dt = datetime.now()
        else:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        if start_date is None:
            start_dt = end_dt - timedelta(days=7)
        else:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')

        # Get detections
        detections_df = storage.get_detections(start_dt, end_dt)

        if detections_df.empty:
            return {"bins": [], "counts": []}

        # Calculate histogram
        import numpy as np
        hist, bin_edges = np.histogram(detections_df['speed_kmh'], bins=bins)

        return {
            "bins": bin_edges.tolist(),
            "counts": hist.tolist()
        }

    except Exception as e:
        logger.error(f"Error getting speed distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats/vehicle_types")
async def get_vehicle_types(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get vehicle type breakdown"""
    try:
        # Parse dates
        if end_date is None:
            end_dt = datetime.now()
        else:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        if start_date is None:
            start_dt = end_dt - timedelta(days=7)
        else:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')

        # Get detections
        detections_df = storage.get_detections(start_dt, end_dt)

        if detections_df.empty:
            return {"vehicle_types": {}}

        # Count by type
        type_counts = detections_df['object_type'].value_counts().to_dict()

        return {"vehicle_types": type_counts}

    except Exception as e:
        logger.error(f"Error getting vehicle types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def generate_video_stream(quality: int = 90, detection_interval: int = 3, enable_detection: bool = True):
    """
    Generate video stream with optional detections

    Args:
        quality: JPEG quality (0-100), higher = better quality
        detection_interval: Run detection every N frames (1 = every frame, 3 = every 3rd frame)
        enable_detection: Enable object detection overlay
    """
    global camera_stream, detector

    if camera_stream is None:
        # Return placeholder image
        import numpy as np
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera not configured", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', placeholder)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    # Connect to camera if not connected
    if not camera_stream.is_connected:
        if not camera_stream.connect():
            logger.error("Failed to connect to camera for streaming")
            return

    # Load detector if needed
    if enable_detection and detector and not detector.is_loaded:
        detector.load_model()

    frame_count = 0
    last_detections = []

    try:
        for frame in camera_stream.frame_generator(auto_reconnect=True):
            frame_count += 1

            # Run detection only every N frames to reduce CPU load
            if enable_detection and detector and detector.is_loaded and (frame_count % detection_interval == 0):
                annotated_frame, last_detections = detector.detect_and_visualize(frame)
            else:
                # Use raw frame or overlay previous detections
                annotated_frame = frame

            # Encode frame as JPEG with high quality
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    except Exception as e:
        logger.error(f"Error in video stream: {e}")


@app.get("/api/video_feed")
async def video_feed(
    quality: int = 90,
    detection_interval: int = 3,
    enable_detection: bool = True
):
    """
    Video feed endpoint with configurable quality and performance

    Args:
        quality: JPEG quality (1-100), default 90. Higher = better quality but larger bandwidth
        detection_interval: Run detection every N frames, default 3. Higher = better performance
        enable_detection: Enable object detection overlay, default True

    Examples:
        /api/video_feed?quality=95&detection_interval=1  # Best quality, all frames detected
        /api/video_feed?quality=90&detection_interval=3  # Balanced (default)
        /api/video_feed?quality=85&detection_interval=5  # Better performance
        /api/video_feed?enable_detection=false           # Raw camera feed, no detection
    """
    # Validate parameters
    quality = max(1, min(100, quality))
    detection_interval = max(1, min(10, detection_interval))

    return StreamingResponse(
        generate_video_stream(quality, detection_interval, enable_detection),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "camera_enabled": camera_stream is not None
    }


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page"""
    return templates.TemplateResponse("settings.html", {"request": request})


@app.get("/api/calibration/capture")
async def capture_calibration_frame():
    """Capture a frame for calibration"""
    global camera_stream

    if camera_stream is None or not camera_stream.is_connected:
        raise HTTPException(status_code=503, detail="Camera not available")

    try:
        frame = camera_stream.get_frame()
        if frame is None:
            raise HTTPException(status_code=500, detail="Failed to capture frame")

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        from fastapi.responses import Response
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Error capturing calibration frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calibration/save")
async def save_calibration(request: Request):
    """Save calibration data"""
    import json
    from datetime import datetime

    try:
        data = await request.json()

        # Validate required fields
        required_fields = ['point1', 'point2', 'real_distance_meters', 'pixel_distance', 'pixels_per_meter']
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing field: {field}")

        # Prepare calibration data
        calibration_data = {
            "point1": data['point1'],
            "point2": data['point2'],
            "real_distance_meters": data['real_distance_meters'],
            "pixel_distance": data['pixel_distance'],
            "pixels_per_meter": data['pixels_per_meter'],
            "calibration_date": datetime.now().isoformat(),
            "notes": "Calibrated via dashboard"
        }

        # Save to file
        calibration_file = Path("calibration.json")
        with open(calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        logger.info(f"Calibration saved: {calibration_data['pixels_per_meter']:.2f} pixels/meter")

        return {
            "status": "success",
            "message": "Calibration saved successfully",
            "pixels_per_meter": calibration_data['pixels_per_meter']
        }

    except Exception as e:
        logger.error(f"Error saving calibration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/calibration/current")
async def get_current_calibration():
    """Get current calibration data"""
    import json

    try:
        calibration_file = Path("calibration.json")
        if not calibration_file.exists():
            return {"calibrated": False}

        with open(calibration_file, 'r') as f:
            data = json.load(f)

        return {
            "calibrated": True,
            **data
        }

    except Exception as e:
        logger.error(f"Error loading calibration: {e}")
        return {"calibrated": False, "error": str(e)}


def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Starting Traffic Monitoring Dashboard...")
    logger.info("Open http://localhost:8000 in your browser")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
