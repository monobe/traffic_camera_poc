# System Design Document

## 1. System Architecture

### 1.1 Overview

The traffic monitoring system follows a **pipeline architecture** with the following stages:

```
Camera → Capture → Detection → Tracking → Speed Estimation → Storage → Reporting
```

### 1.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Traffic Monitoring System                │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  IP Camera   │────▶│   Capture    │────▶│  Detection   │
│ (Tapo C320WS)│RTSP │   Module     │Frame│  (YOLOv8)    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
                                                   │ Detections
                                                   ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Reporting  │◀────│   Storage    │◀────│   Tracking   │
│   Module     │     │   Module     │     │  (ByteTrack) │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                      │
       │                    │                      ▼
       ▼                    ▼              ┌──────────────┐
┌──────────────┐     ┌──────────────┐     │    Speed     │
│  PDF Report  │     │SQLite + CSV  │     │  Estimation  │
└──────────────┘     └──────────────┘     └──────────────┘

Optional:
┌──────────────┐
│  Dashboard   │
│  (FastAPI +  │◀─── SQLite (read-only)
│   React)     │
└──────────────┘
```

### 1.3 Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│             Presentation Layer                       │
│  - Dashboard UI                                      │
│  - PDF Report Generator                              │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│             Application Layer                        │
│  - Traffic Analyzer                                  │
│  - Statistics Aggregator                             │
│  - Report Builder                                    │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│             Domain Layer                             │
│  - Speed Estimator                                   │
│  - Object Tracker                                    │
│  - Calibration Engine                                │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│             Infrastructure Layer                     │
│  - Camera Capture                                    │
│  - YOLO Detector                                     │
│  - Database (SQLite)                                 │
│  - File System (CSV)                                 │
└─────────────────────────────────────────────────────┘
```

## 2. Module Design

### 2.1 Module Overview

| Module | Responsibility | Input | Output |
|--------|---------------|-------|--------|
| **capture** | Capture video stream from IP camera | RTSP URL | Frame iterator |
| **detection** | Detect objects in frames | Frame | List of bounding boxes |
| **tracking** | Track objects across frames | Detections | Track IDs + trajectories |
| **calibration** | Camera calibration for distance | User input | Calibration config |
| **speed_estimation** | Calculate speed from trajectories | Tracks + calibration | Speed (km/h) |
| **stats** | Aggregate statistics | Detection events | Hourly/daily stats |
| **report** | Generate PDF reports | Statistics | PDF file |
| **dashboard** | Web UI for monitoring | Database | HTML/JSON |

### 2.2 Module Details

#### 2.2.1 Capture Module

**Purpose**: Capture video frames from IP camera

```python
# capture/stream.py

class CameraStream:
    """RTSP stream handler"""

    def __init__(self, rtsp_url: str, fps: int = 15):
        self.rtsp_url = rtsp_url
        self.fps = fps
        self.cap = None

    def connect(self) -> bool:
        """Connect to camera"""

    def get_frame(self) -> Optional[np.ndarray]:
        """Get single frame"""

    def frame_generator(self) -> Iterator[np.ndarray]:
        """Yield frames continuously"""

    def release(self):
        """Release camera connection"""
```

**Dependencies**:
- OpenCV (cv2)
- FFmpeg

**Configuration**:
```yaml
camera:
  rtsp_url: "rtsp://user:pass@192.168.1.100:554/stream1"
  fps: 15
  resolution: [1920, 1080]
  reconnect_timeout: 30
```

#### 2.2.2 Detection Module

**Purpose**: Detect vehicles and pedestrians using YOLO

```python
# detection/detector.py

class YOLODetector:
    """YOLOv8 object detector"""

    def __init__(self, model_path: str, confidence: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.classes = [2, 3, 5, 7]  # car, motorcycle, bus, bicycle

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in frame"""

    def filter_classes(self, results) -> List[Detection]:
        """Filter by vehicle classes"""

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    timestamp: datetime
```

**Dependencies**:
- ultralytics (YOLOv8)
- PyTorch (CPU version)

**Configuration**:
```yaml
detection:
  model: "yolov8n.pt"  # yolov8n, yolov8s, yolov8m
  confidence: 0.5
  classes: [2, 3, 5, 7]  # COCO classes: car, motorcycle, bus, bicycle
  device: "cpu"
```

#### 2.2.3 Tracking Module

**Purpose**: Track objects across frames

```python
# tracking/tracker.py

class ObjectTracker:
    """Multi-object tracker using ByteTrack or SORT"""

    def __init__(self, algorithm: str = "bytetrack"):
        self.algorithm = algorithm
        self.tracker = self._init_tracker()
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracks with new detections"""

    def get_active_tracks(self) -> List[Track]:
        """Get currently active tracks"""

    def remove_lost_tracks(self, max_age: int = 30):
        """Remove tracks not updated for max_age frames"""

@dataclass
class Track:
    track_id: int
    class_name: str
    trajectory: List[Tuple[int, int]]  # (x, y) center points
    timestamps: List[datetime]
    bbox_history: List[Tuple[int, int, int, int]]
    confidence_history: List[float]
    state: str  # 'active', 'lost', 'finished'
```

**Dependencies**:
- ByteTrack or SORT implementation

**Configuration**:
```yaml
tracking:
  algorithm: "bytetrack"  # or "sort"
  max_age: 30  # frames
  min_hits: 3  # minimum detections to start track
  iou_threshold: 0.3
```

#### 2.2.4 Calibration Module

**Purpose**: Calibrate camera for distance measurement

```python
# calibration/calibrate.py

class CameraCalibrator:
    """Camera calibration for speed measurement"""

    def __init__(self):
        self.points: List[Tuple[int, int]] = []
        self.real_distance: Optional[float] = None
        self.pixels_per_meter: Optional[float] = None

    def add_point(self, x: int, y: int):
        """Add calibration point"""

    def set_real_distance(self, distance_meters: float):
        """Set real-world distance between points"""

    def calculate_scale(self) -> float:
        """Calculate pixels per meter"""

    def save(self, filepath: str):
        """Save calibration to JSON"""

    def load(self, filepath: str):
        """Load calibration from JSON"""

@dataclass
class CalibrationData:
    point1: Tuple[int, int]
    point2: Tuple[int, int]
    real_distance_meters: float
    pixels_per_meter: float
    calibration_date: datetime
```

**Output Format** (calibration.json):
```json
{
  "point1": [100, 200],
  "point2": [500, 200],
  "real_distance_meters": 10.0,
  "pixels_per_meter": 40.0,
  "calibration_date": "2024-01-15T10:30:00"
}
```

#### 2.2.5 Speed Estimation Module

**Purpose**: Calculate vehicle speed from trajectories

```python
# speed_estimation/estimator.py

class SpeedEstimator:
    """Estimate speed from object trajectories"""

    def __init__(self, calibration: CalibrationData, fps: int):
        self.calibration = calibration
        self.fps = fps
        self.min_track_length = 10

    def estimate_speed(self, track: Track) -> Optional[SpeedEstimate]:
        """Calculate speed from trajectory"""
        # Formula: distance (meters) / time (seconds) * 3.6 = km/h

    def get_direction(self, track: Track) -> str:
        """Determine movement direction (uphill/downhill)"""

@dataclass
class SpeedEstimate:
    track_id: int
    object_type: str
    speed_kmh: float
    direction: str  # 'uphill', 'downhill'
    confidence: float
    timestamp: datetime
    trajectory_length: int
```

**Algorithm**:
```python
def calculate_speed(trajectory, timestamps, pixels_per_meter, fps):
    # Get start and end points
    start_point = trajectory[0]
    end_point = trajectory[-1]

    # Calculate pixel distance
    pixel_distance = euclidean_distance(start_point, end_point)

    # Convert to meters
    distance_meters = pixel_distance / pixels_per_meter

    # Calculate time
    time_seconds = len(trajectory) / fps

    # Calculate speed (m/s)
    speed_ms = distance_meters / time_seconds

    # Convert to km/h
    speed_kmh = speed_ms * 3.6

    return speed_kmh
```

#### 2.2.6 Storage Module

**Purpose**: Store detection events and statistics

```python
# storage/database.py

class StorageManager:
    """Manage SQLite database and CSV exports"""

    def __init__(self, db_path: str, csv_dir: str):
        self.db_path = db_path
        self.csv_dir = csv_dir
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def save_detection(self, speed_estimate: SpeedEstimate):
        """Save detection event"""

    def save_csv(self, speed_estimate: SpeedEstimate):
        """Append to daily CSV file"""

    def get_hourly_stats(self, date: datetime) -> pd.DataFrame:
        """Get hourly aggregated statistics"""

    def get_daily_summary(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get summary statistics for date range"""
```

**Database Schema**:
```sql
-- detections table
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    track_id INTEGER NOT NULL,
    object_type VARCHAR(20) NOT NULL,
    speed_kmh REAL NOT NULL,
    direction VARCHAR(10),
    confidence REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_timestamp ON detections(timestamp);
CREATE INDEX idx_object_type ON detections(object_type);

-- hourly_stats table
CREATE TABLE hourly_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    hour INTEGER NOT NULL,
    object_type VARCHAR(20) NOT NULL,
    count INTEGER NOT NULL,
    avg_speed REAL,
    max_speed REAL,
    speeding_count INTEGER,  -- over 30 km/h
    UNIQUE(date, hour, object_type)
);

-- daily_stats table
CREATE TABLE daily_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL UNIQUE,
    total_vehicles INTEGER,
    avg_speed REAL,
    max_speed REAL,
    speeding_count INTEGER,
    speeding_rate REAL,
    peak_hour INTEGER
);
```

**CSV Format**:
```csv
timestamp,track_id,object_type,speed_kmh,direction,confidence
2024-01-15 08:15:32,1,car,45.2,downhill,0.92
2024-01-15 08:16:10,2,bicycle,28.5,downhill,0.87
```

#### 2.2.7 Report Module

**Purpose**: Generate PDF reports

```python
# report/generator.py

class ReportGenerator:
    """Generate PDF reports for authorities"""

    def __init__(self, storage: StorageManager):
        self.storage = storage
        self.template = "police_report"

    def generate_report(
        self,
        start_date: datetime,
        end_date: datetime,
        output_path: str
    ):
        """Generate PDF report"""

    def _create_charts(self, data: pd.DataFrame) -> List[str]:
        """Generate charts using matplotlib"""

    def _build_pdf(self, data: Dict, charts: List[str], output: str):
        """Build PDF using reportlab"""
```

**Report Structure**:
1. Header (Recipient, Title, Date)
2. Survey Overview
3. Key Findings (table)
4. Speed Distribution Chart
5. Hourly Traffic Volume Chart
6. Speeding Rate Chart
7. Recommendations
8. Appendix (Methodology, Error Margins)

#### 2.2.8 Dashboard Module (Optional)

**Purpose**: Web-based monitoring interface

```python
# dashboard/app.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

@app.get("/api/realtime")
async def get_realtime_data():
    """Get latest detection data"""

@app.get("/api/stats/hourly")
async def get_hourly_stats(date: str):
    """Get hourly statistics"""

@app.get("/api/stats/daily")
async def get_daily_stats(start_date: str, end_date: str):
    """Get daily statistics"""
```

## 3. Data Flow

### 3.1 Real-time Processing Pipeline

```
1. Camera Stream (RTSP)
   ↓
2. Frame Capture (15 FPS)
   ↓
3. YOLO Detection
   ↓ List[Detection]
4. Object Tracking (ByteTrack)
   ↓ List[Track]
5. Speed Estimation
   ↓ List[SpeedEstimate]
6. Storage (SQLite + CSV)
   ↓
7. Statistics Aggregation (hourly batch)
```

### 3.2 Report Generation Flow

```
1. Query Date Range
   ↓
2. Load Data from SQLite
   ↓
3. Calculate Statistics
   ↓
4. Generate Charts (matplotlib)
   ↓
5. Build PDF (reportlab)
   ↓
6. Save to reports/
```

## 4. Configuration Management

### 4.1 Configuration File Structure

**config.yaml**:
```yaml
# Camera settings
camera:
  rtsp_url: "rtsp://admin:password@192.168.1.100:554/stream1"
  fps: 15
  resolution: [1920, 1080]
  reconnect_timeout: 30

# Detection settings
detection:
  model: "yolov8n.pt"
  confidence: 0.5
  classes: [2, 3, 5, 7]  # car, motorcycle, bus, bicycle
  device: "cpu"

# Tracking settings
tracking:
  algorithm: "bytetrack"
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3

# Speed estimation
speed_estimation:
  calibration_file: "calibration.json"
  min_track_length: 10
  speed_limit_kmh: 30

# Storage
storage:
  database: "./data/traffic.db"
  csv_dir: "./data/csv"
  retention_days: 90

# Reporting
reporting:
  output_dir: "./reports"
  location: "横浜市神奈川区・三ツ沢第25号線"
  recipient: "神奈川警察署 交通課"

# Dashboard (optional)
dashboard:
  enabled: false
  host: "0.0.0.0"
  port: 8000

# Logging
logging:
  level: "INFO"
  file: "./logs/traffic_monitor.log"
  max_size_mb: 100
  backup_count: 5
```

### 4.2 Configuration Loading

```python
# config/loader.py

import yaml
from typing import Dict, Any

class Config:
    """Configuration manager"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self):
        """Load configuration from YAML"""
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value
```

## 5. Error Handling & Logging

### 5.1 Error Handling Strategy

```python
# Common exceptions

class TrafficMonitorError(Exception):
    """Base exception"""

class CameraConnectionError(TrafficMonitorError):
    """Camera connection failed"""

class CalibrationError(TrafficMonitorError):
    """Calibration invalid or missing"""

class DetectionError(TrafficMonitorError):
    """Detection failed"""
```

### 5.2 Logging Strategy

```python
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_monitor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

**Log Levels**:
- **DEBUG**: Frame-by-frame processing details
- **INFO**: Detection events, statistics updates
- **WARNING**: Tracking lost, low confidence detections
- **ERROR**: Camera disconnection, processing failures
- **CRITICAL**: System failures

## 6. Performance Considerations

### 6.1 Optimization Strategies

1. **Frame Skipping**: Process every Nth frame if CPU is overloaded
2. **Batch Processing**: Process video files in batch mode
3. **Model Selection**: Use YOLOv8n for real-time, YOLOv8s for accuracy
4. **Database Indexing**: Index by timestamp and object_type
5. **Async I/O**: Use async for camera capture and storage

### 6.2 Resource Requirements

| Component | CPU Usage | Memory |
|-----------|-----------|--------|
| Camera Capture | 5-10% | 100 MB |
| YOLOv8n Detection | 30-50% | 500 MB |
| Tracking | 5-10% | 100 MB |
| Total | 40-70% | ~700 MB |

**Expected Performance**:
- **FPS**: 10-15 (real-time on modern CPU)
- **Latency**: <100ms per frame
- **Storage**: ~1 MB/hour (CSV), ~10 MB/day (SQLite)

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# tests/test_speed_estimator.py

def test_speed_calculation():
    """Test speed calculation accuracy"""

def test_direction_detection():
    """Test uphill/downhill detection"""
```

### 7.2 Integration Tests

```python
# tests/test_pipeline.py

def test_end_to_end_pipeline():
    """Test full pipeline with sample video"""
```

### 7.3 Calibration Validation

- Test with known-speed vehicles
- Compare with radar gun measurements
- Validate ±5 km/h accuracy

## 8. Deployment

### 8.1 System Setup

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg sqlite3

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 8.2 Camera Setup

1. Mount camera at 3-4m height
2. Configure RTSP stream
3. Test connection: `ffplay rtsp://...`

### 8.3 Calibration

```bash
python calibration/calibrate.py --camera-url "rtsp://..."
```

### 8.4 Running

```bash
# Run monitoring
python main.py --config config.yaml

# Generate report
python report/generate_pdf.py --start-date 2024-01-01 --end-date 2024-01-07
```

### 8.5 Systemd Service (Optional)

```ini
# /etc/systemd/system/traffic-monitor.service

[Unit]
Description=Traffic Monitoring Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/traffic_camera_poc
Environment="PATH=/home/ubuntu/traffic_camera_poc/venv/bin"
ExecStart=/home/ubuntu/traffic_camera_poc/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## 9. Future Enhancements

### Phase 2: Advanced Features
- Vehicle type classification (car, taxi, bus)
- License plate detection (with privacy protection)
- Bicycle wrong-way detection
- Dangerous behavior detection

### Phase 3: Cloud Integration
- AWS S3 / Cloudflare R2 backup
- Multi-location monitoring
- Cloud dashboard
- Mobile app

### Phase 4: AI Improvements
- Fine-tuned YOLO model for local traffic
- Weather condition detection
- Predictive analytics

## 10. Security & Privacy

### 10.1 Privacy Considerations

- No license plate storage (blur if needed)
- No facial recognition
- Data retention policy (90 days default)
- Anonymized statistics only in reports

### 10.2 Security Measures

- Secure RTSP credentials in environment variables
- Database access control
- Regular backups
- HTTPS for dashboard (if deployed)

---

**Document Version**: 1.0
**Last Updated**: 2024-01-15
**Author**: Traffic Camera PoC Team
