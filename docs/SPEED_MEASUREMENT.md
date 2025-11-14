# Speed Measurement Guidelines

## Camera Setup for Accurate Speed Measurement

### Critical Requirements

To accurately measure vehicle speed at high speeds (50-60 km/h), you need **sufficient monitoring range** and **adequate frame rate**.

### Problem: Short Monitoring Range

**Example calculation for 5m monitoring range:**

```
Vehicle speed: 60 km/h = 16.7 m/s
Time to pass 5m: 5m / 16.7 m/s = 0.30 seconds

With 15 FPS:
Frames captured: 15 FPS × 0.30s = 4.5 frames

With min_track_length = 10:
Required frames: 10 frames
Result: INSUFFICIENT - Cannot measure speed!
```

### Recommended Setup

#### Option 1: Wide Monitoring Range (BEST)

**15-20m monitoring range:**

```
Vehicle speed: 60 km/h = 16.7 m/s
Time to pass 15m: 15m / 16.7 m/s = 0.90 seconds

With 20 FPS:
Frames captured: 20 FPS × 0.90s = 18 frames ✓

With min_track_length = 7:
Required frames: 7 frames ✓
Result: SUFFICIENT - Good accuracy!
```

**Camera requirements:**
- Wide angle lens to cover 15-20m road section
- Mounted 3-4m above ground
- Perpendicular to traffic flow

#### Option 2: Higher Frame Rate (if range is limited)

If you can only monitor 5-8m range:

```yaml
camera:
  fps: 30  # Increased from 15 to 30

speed_estimation:
  min_track_length: 7  # Reduced from 10
```

**Trade-offs:**
- ✓ Works with shorter monitoring range
- ✗ Higher CPU load (2× processing)
- ✗ Lower effective FPS due to YOLO bottleneck

### Recommended Configuration

**For residential streets (30-60 km/h traffic):**

```yaml
camera:
  fps: 20  # Good balance
  resolution: [1920, 1080]

speed_estimation:
  min_track_length: 7  # 0.35 seconds at 20 FPS
  # Monitoring range: 15-20m recommended
```

**Minimum viable setup:**
- Monitoring range: 15m
- FPS: 20
- Min track length: 7 frames
- Expected capture time: 0.35-0.90 seconds

### Speed Accuracy by Range

| Range | 30 km/h | 50 km/h | 60 km/h | Frames @20FPS | Accuracy |
|-------|---------|---------|---------|---------------|----------|
| 5m    | 0.60s   | 0.36s   | 0.30s   | 6-12 frames   | Poor     |
| 10m   | 1.20s   | 0.72s   | 0.60s   | 12-24 frames  | Fair     |
| 15m   | 1.80s   | 1.08s   | 0.90s   | 18-36 frames  | Good     |
| 20m   | 2.40s   | 1.44s   | 1.20s   | 24-48 frames  | Excellent|

**Recommendation: Use 15-20m range for reliable measurements**

### Camera Placement Tips

1. **Height**: 3-4 meters above ground
2. **Angle**: Perpendicular to road (minimize perspective distortion)
3. **Coverage**: Ensure 15-20m continuous road section is visible
4. **Lighting**: Good lighting or use camera with night vision
5. **Stability**: Secure mounting (no vibration/movement)

### Calibration

Use Google Maps satellite view to measure exact distance:

1. Identify two clear reference points (e.g., road markings, poles)
2. Measure distance using Maps ruler tool
3. Use dashboard calibration tool to mark points
4. System calculates pixels_per_meter automatically

### Testing Speed Accuracy

**Method 1: Known speed vehicle**
- Drive at known speed (GPS speedometer)
- Compare with system measurement
- Adjust if error > ±5 km/h

**Method 2: Two-point timing**
- Manually time vehicle between two points
- Calculate speed: distance / time × 3.6
- Compare with system measurement

### Common Issues

**Issue**: Speed always shows as 0 or very low
- **Cause**: Monitoring range too short
- **Solution**: Widen camera angle or increase FPS

**Issue**: Inconsistent speed readings
- **Cause**: Poor calibration or camera movement
- **Solution**: Re-calibrate, secure camera mount

**Issue**: Missing high-speed vehicles
- **Cause**: Insufficient frame rate
- **Solution**: Increase FPS to 20-30

### Performance Considerations

**CPU Load by FPS:**
- 15 FPS: ~70% CPU (YOLOv8n on modern CPU)
- 20 FPS: ~90% CPU
- 30 FPS: CPU may not keep up → frame drops

**GPU Acceleration:**
If available, enable GPU in config.yaml:

```yaml
detection:
  device: "cuda"  # Requires NVIDIA GPU
```

With GPU:
- 15 FPS: ~10% GPU, smooth operation
- 30 FPS: ~20% GPU, excellent performance
- 60 FPS: Possible with dedicated GPU

### Summary

**Minimum Requirements for 50-60 km/h measurement:**
- ✓ Monitoring range: 15m or more
- ✓ Frame rate: 20 FPS
- ✓ Min track length: 7 frames
- ✓ Stable camera mounting
- ✓ Proper calibration

**For best results:**
- Use 20m monitoring range
- 20-30 FPS
- GPU acceleration if available
- Regular calibration checks
