# Dashboard Specification

## Overview

Web-based real-time traffic monitoring dashboard for viewing detection results, statistics, and historical data.

## Technology Stack

- **Backend**: FastAPI (Python web framework)
- **Frontend**: HTML + JavaScript (Vanilla JS with Chart.js)
- **WebSocket**: Real-time updates
- **Database**: SQLite (via StorageManager)

## Features

### 1. Real-time Monitoring

**Live Detection Feed**:
- Current active tracks
- Recent detections (last 10)
- Live statistics update

**Real-time Metrics**:
- Vehicles per minute
- Average speed
- Speeding count
- Active tracks count

### 2. Historical Data View

**Date Range Selector**:
- Select start and end dates
- Quick presets (Today, Yesterday, Last 7 days, Last 30 days)

**Statistics Display**:
- Total vehicles
- Average/Max/Min speed
- Speeding rate
- Peak hours

### 3. Charts and Visualizations

**Speed Distribution**:
- Histogram of speed measurements
- Speed limit indicator line

**Hourly Traffic Volume**:
- Bar chart of vehicles per hour
- Peak time highlighting

**Speeding Rate Timeline**:
- Line chart showing speeding rate over time
- By hour or by day

**Vehicle Type Breakdown**:
- Pie chart of detected vehicle types
- Car, motorcycle, bicycle, etc.

### 4. Data Export

**CSV Export**:
- Export filtered data to CSV
- Daily summaries

**PDF Report**:
- Generate PDF report for selected date range
- Download directly from dashboard

## API Endpoints

### Real-time Data

```
GET /api/realtime
Response: {
  "active_tracks": 5,
  "recent_detections": [...],
  "current_stats": {
    "vehicles_per_minute": 12.5,
    "avg_speed": 42.3,
    "speeding_count": 3
  }
}
```

### Historical Stats

```
GET /api/stats/daily?start_date=2024-01-01&end_date=2024-01-07
Response: {
  "summary": {...},
  "daily_data": [...]
}
```

### Hourly Stats

```
GET /api/stats/hourly?date=2024-01-15
Response: {
  "hourly_data": [...]
}
```

### Speed Distribution

```
GET /api/stats/speed_distribution?start_date=2024-01-01&end_date=2024-01-07
Response: {
  "bins": [0, 10, 20, 30, 40, 50, 60],
  "counts": [5, 20, 45, 30, 15, 5, 2]
}
```

### Export Data

```
POST /api/export/csv
Body: {
  "start_date": "2024-01-01",
  "end_date": "2024-01-07"
}
Response: CSV file download
```

```
POST /api/export/pdf
Body: {
  "start_date": "2024-01-01",
  "end_date": "2024-01-07"
}
Response: PDF file download
```

## UI Design

### Layout

```
+----------------------------------------------------------+
|  Traffic Monitoring Dashboard            [Date Selector] |
+----------------------------------------------------------+
|                                                           |
|  +-- Real-time Stats --+  +-- Recent Detections -----+  |
|  | Active Tracks: 5    |  | Track #123 | Car | 45km/h|  |
|  | Vehicles/min: 12.5  |  | Track #124 | Bike| 28km/h|  |
|  | Avg Speed: 42.3 km/h|  | Track #125 | Car | 52km/h|  |
|  | Speeding: 3         |  | ...                       |  |
|  +---------------------+  +---------------------------+  |
|                                                           |
|  +-- Speed Distribution --------------------------------+ |
|  |                                                       | |
|  |     [Histogram Chart]                                | |
|  |                                                       | |
|  +-------------------------------------------------------+ |
|                                                           |
|  +-- Hourly Traffic ---+  +-- Vehicle Types ----------+ |
|  |                     |  |                           | |
|  | [Bar Chart]         |  | [Pie Chart]               | |
|  |                     |  |                           | |
|  +---------------------+  +---------------------------+ |
|                                                           |
|  +-- Export Options ------------------------------------+ |
|  | [Export CSV] [Generate PDF Report]                   | |
|  +-------------------------------------------------------+ |
+----------------------------------------------------------+
```

### Color Scheme

- **Normal Speed**: Green (#4CAF50)
- **Slight Over**: Yellow (#FFC107)
- **Speeding**: Orange (#FF9800)
- **Significant Over**: Red (#F44336)

### Responsive Design

- Desktop: Full layout with all panels
- Tablet: Stacked panels, 2 columns
- Mobile: Single column, collapsible panels

## Configuration

```yaml
dashboard:
  enabled: true
  host: "0.0.0.0"
  port: 8000
  debug: false
  refresh_interval: 5  # seconds
  max_recent_detections: 20
```

## Security

- **Access Control**: Basic authentication (optional)
- **CORS**: Configure allowed origins
- **Rate Limiting**: Prevent API abuse
- **Data Privacy**: No personal information displayed

## Performance

- **Caching**: Cache historical stats for 5 minutes
- **Pagination**: Limit API response sizes
- **WebSocket**: Use for real-time updates instead of polling
- **Database Indexing**: Ensure proper indexes for fast queries

## Future Enhancements

### Phase 1 (Current)
- Basic real-time monitoring
- Historical data view
- Static charts
- CSV/PDF export

### Phase 2
- Live video feed display
- Interactive charts (zoom, pan)
- User authentication
- Multiple camera support

### Phase 3
- Mobile app
- Email/SMS alerts for speeding
- Integration with external systems
- Advanced analytics (ML predictions)

## Testing

- **Unit Tests**: API endpoint tests
- **Integration Tests**: End-to-end flow tests
- **Load Tests**: Handle 100+ concurrent users
- **Browser Compatibility**: Chrome, Firefox, Safari, Edge

## Deployment

### Development
```bash
python dashboard/app.py
```

### Production
```bash
# Using Gunicorn
gunicorn dashboard.app:app --workers 4 --bind 0.0.0.0:8000

# Using systemd service
systemctl start traffic-dashboard
```

### Docker (Future)
```bash
docker build -t traffic-dashboard .
docker run -p 8000:8000 traffic-dashboard
```

## Dependencies

- fastapi
- uvicorn (ASGI server)
- jinja2 (templating)
- python-multipart (file uploads)
- Chart.js (frontend charting)

---

**Version**: 1.0
**Last Updated**: 2024-11-14
