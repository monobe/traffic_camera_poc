# Traffic Camera PoC

AI-powered traffic monitoring system for residential areas, school zones, and sloped roads.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This project monitors vehicle speed and traffic volume in residential areas using computer vision and AI. It automatically generates reports suitable for submission to police departments and local government offices.

**Target Use Case**: Yokohama City, Kanagawa Ward - Sloped residential streets with speeding issues and school zones.

## Features

- **Object Detection**: Detect cars, motorcycles, bicycles, and pedestrians using YOLOv8/v11
- **Vehicle Tracking**: Track individual vehicles across frames using ByteTrack/SORT
- **Speed Estimation**: Calculate vehicle speed from single camera footage
- **Traffic Volume Analysis**: Hourly, daily, and weekly traffic statistics
- **Automated PDF Reports**: Generate professional reports for authorities
- **Local Dashboard**: Web-based interface for real-time monitoring (optional)

## System Requirements

### Hardware

| Component | Requirement |
|-----------|-------------|
| **PC** | Ubuntu 20.04+ (Windows laptop converted to Ubuntu is OK) |
| **CPU** | Multi-core processor (no GPU required for YOLOv8n/s) |
| **RAM** | 4GB+ recommended |
| **Camera** | TP-Link Tapo C320WS (2K QHD, Color Night Vision) |
| **Network** | WiFi connection |
| **Power** | AC power outlet |

### Software

- Python 3.8+
- OpenCV
- YOLOv8 (Ultralytics)
- ByteTrack or SORT
- FFmpeg (for camera stream capture)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/monobe/traffic_camera_poc.git
cd traffic_camera_poc
```

### 2. Install System Dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg
```

### 3. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Python Packages

```bash
pip install -r requirements.txt
```

## Project Structure

```
traffic-monitoring/
├── capture/            # Camera stream capture
├── detection/          # YOLOv8 object detection
├── tracking/           # ByteTrack/SORT tracking
├── calibration/        # Distance calibration tools
├── speed_estimation/   # Speed calculation
├── stats/              # Daily statistics
├── report/             # PDF report generation
├── dashboard/          # Web UI (optional)
├── config.yaml         # Configuration file
├── requirements.txt    # Python dependencies
└── README.md
```

## Quick Start

### 1. Camera Setup

Configure your TP-Link Tapo C320WS camera:

1. Install camera and connect to power
2. Set up camera via Tapo mobile app
3. Enable RTSP stream in camera settings
4. Note down RTSP URL: `rtsp://username:password@camera_ip:554/stream1`

### 2. Calibration

Before running speed estimation, calibrate the camera view:

```bash
python calibration/calibrate.py --camera-url "rtsp://..." --output calibration.json
```

Follow on-screen instructions to mark two points of known distance on the road.

### 3. Configuration

Edit `config.yaml`:

```yaml
camera:
  rtsp_url: "rtsp://username:password@192.168.1.100:554/stream1"
  fps: 15
  resolution: [1920, 1080]

detection:
  model: "yolov8n.pt"  # or yolov8s.pt for better accuracy
  confidence: 0.5
  classes: [2, 3, 5, 7]  # car, motorcycle, bus, bicycle

tracking:
  algorithm: "bytetrack"  # or "sort"

speed_estimation:
  calibration_file: "calibration.json"
  min_track_length: 10  # minimum frames to estimate speed

output:
  data_dir: "./data"
  report_dir: "./reports"

logging:
  level: "INFO"
  file: "traffic_monitor.log"
```

### 4. Run Monitoring

```bash
python main.py --config config.yaml
```

### 5. Generate Report

```bash
python report/generate_pdf.py --start-date 2024-01-01 --end-date 2024-01-07 --output weekly_report.pdf
```

## Output Data

### CSV Format

Data is continuously logged to CSV files:

```csv
timestamp,object_type,speed_kmh,direction,confidence,track_id
2024-01-15 08:15:32,car,45.2,downhill,0.92,1
2024-01-15 08:16:10,bicycle,28.5,downhill,0.87,2
2024-01-15 08:17:05,car,52.1,uphill,0.95,3
```

### SQLite Database

Statistics are stored in SQLite for efficient querying:

- `detections` - All detection events
- `hourly_stats` - Hourly aggregated traffic volume
- `daily_stats` - Daily summary statistics

## Report Template

Reports are automatically generated in PDF format suitable for submission to:

- **Kanagawa Police Department (Traffic Division)**
- **Yokohama City Kanagawa Ward Office**

Report includes:

1. **Survey Overview** - Period, location, methodology
2. **Key Findings** - Average speed, speeding rate, peak hours
3. **Charts & Graphs** - Speed histogram, hourly traffic volume
4. **Recommendations** - Zone30 review, speed reduction measures
5. **Appendix** - Methodology, error margins, sample images

## Dashboard (Optional)

Start the web dashboard:

```bash
python dashboard/app.py
```

Access at `http://localhost:8000`

Features:
- Real-time traffic monitoring
- Speed distribution charts
- Heatmap of traffic volume
- Daily/weekly comparisons

## Configuration Tips

### Camera Placement

- **Height**: 3-4 meters above ground
- **Angle**: Perpendicular to road for accurate speed measurement
- **Coverage**: Ensure clear view of 10-20m road section
- **Weather Protection**: Use weatherproof housing if needed

### Speed Accuracy

- Calibrate using Google Maps for exact distances
- Ensure camera is securely mounted (no vibration)
- Test with known speed vehicles
- Expected accuracy: ±5 km/h

### Performance Optimization

- Use YOLOv8n for CPU-only systems
- Reduce FPS if CPU usage is high (10-15 FPS is sufficient)
- Process in batch mode for historical footage

## Roadmap

### Phase 1: MVP (Current)
- ✅ Basic object detection
- ✅ Speed estimation
- ✅ PDF report generation

### Phase 2: Enhanced Analysis
- [ ] Vehicle type classification (taxi, bus, etc.)
- [ ] Bicycle wrong-way detection
- [ ] Dangerous behavior detection

### Phase 3: Data Integration
- [ ] Cloud sync (AWS S3 / Cloudflare R2)
- [ ] Multi-location analysis
- [ ] Cloud-based dashboard

### Phase 4: OSS Distribution
- [ ] Complete documentation
- [ ] Generalization for other regions
- [ ] Community contributions
- [ ] Deployment guides for municipalities

## Contributing

Contributions are welcome! This project aims to be reusable by communities across Japan facing similar traffic issues.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **YOLOv8** by Ultralytics
- **ByteTrack** for multi-object tracking
- Local community members who supported this initiative

## Contact

For questions or collaboration:

- GitHub Issues: [traffic_camera_poc/issues](https://github.com/monobe/traffic_camera_poc/issues)
- Email: (Add your contact if desired)

## Related Links

- [Project Specification](spec.md)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ByteTrack GitHub](https://github.com/ifzhang/ByteTrack)
- [OpenCV Documentation](https://docs.opencv.org/)

---

**Note**: This system is designed for community safety monitoring. Please respect privacy laws and regulations when deploying camera systems. Consult with local authorities before installation.
