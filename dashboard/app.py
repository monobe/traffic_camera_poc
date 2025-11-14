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
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import application modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from storage.database import StorageManager


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

# Initialize storage
storage = None


def init_storage(db_path: str = "./data/traffic.db", csv_dir: str = "./data/csv"):
    """Initialize storage manager"""
    global storage
    storage = StorageManager(db_path=db_path, csv_dir=csv_dir)
    logger.info("Storage manager initialized")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    init_storage()
    logger.info("Dashboard started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if storage:
        storage.close()
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


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }


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
