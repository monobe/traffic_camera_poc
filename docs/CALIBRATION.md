# Camera Calibration Guide

## Overview

Camera calibration is essential for accurate speed measurement. This process converts pixel distances in the camera image to real-world distances in meters.

## Prerequisites

- Camera must be mounted in final position
- Camera view must be stable (no movement or vibration)
- Clear view of the road

## Calibration Process

### Method 1: Interactive Tool (Recommended)

```bash
source venv/bin/activate
python calibration/calibrate.py --config config.yaml
```

**Steps**:
1. A frame from your camera will be captured
2. Click two points on the road with known distance
3. Enter the real-world distance between these points
4. Calibration data will be saved to `calibration.json`

**Tips for selecting points**:
- Choose clearly visible landmarks (lane markers, road signs, etc.)
- Points should be parallel to vehicle movement direction
- Longer distance = better accuracy (10-20 meters recommended)
- Avoid perspective distortion (don't use points too close to camera)

### Method 2: Manual Calibration

If the GUI tool doesn't work on your system:

1. **Capture a frame**:
   ```bash
   python test_camera.py --config config.yaml
   # Press 's' to save snapshot
   ```

2. **Measure on Google Maps**:
   - Open Google Maps satellite view
   - Zoom to your location
   - Use the "Measure distance" tool
   - Measure a clearly visible section of road

3. **Mark points on image**:
   - Open the snapshot in an image editor
   - Note the pixel coordinates of two points
   - Calculate pixel distance: `sqrt((x2-x1)^2 + (y2-y1)^2)`

4. **Create calibration file manually**:
   ```json
   {
     "point1": [100, 200],
     "point2": [500, 200],
     "real_distance_meters": 10.0,
     "pixel_distance": 400.0,
     "pixels_per_meter": 40.0,
     "calibration_date": "2024-11-14T15:00:00",
     "notes": "Manual calibration"
   }
   ```

## Verification

After calibration, verify accuracy:

```bash
# Check calibration file
cat calibration.json

# Test with known vehicle speed (if possible)
python main.py --config config.yaml
```

## Common Issues

### Issue: Points are hard to see

**Solution**:
- Wait for good lighting conditions
- Use high-contrast landmarks
- Zoom in on the image

### Issue: Inaccurate speed measurements

**Possible causes**:
- Camera moved after calibration → Re-calibrate
- Points too close together → Use longer distance
- Perspective distortion → Choose points in the middle of the frame

### Issue: GUI doesn't work (macOS/headless)

**Solution**: Use Method 2 (Manual Calibration) above

## Camera Placement Tips

For best calibration accuracy:

1. **Height**: 3-4 meters above ground
2. **Angle**: Perpendicular to road (90 degrees)
3. **Coverage**: 10-20 meters of road visible
4. **Stability**: Securely mounted, no vibration

## Calibration Quality

**Good calibration**:
- Points 10-20 meters apart
- Clear, unambiguous landmarks
- Perpendicular to traffic flow
- Middle of camera frame

**Poor calibration**:
- Points too close (<5 meters)
- Unclear or moving reference points
- Strong perspective distortion
- Edge of camera frame

## Re-calibration

Re-calibrate if:
- Camera is moved or adjusted
- Speed measurements seem inaccurate
- After significant time (monthly recommended)
- Weather/seasonal changes affect view

## Example

**Good calibration points**:
- Two white lane markers (10 meters apart)
- Road signs at known distances
- Painted crosswalk edges

**Measurement**:
```
Point 1: Lane marker at (320, 450)
Point 2: Lane marker at (780, 450)
Pixel distance: 460 pixels
Real distance: 10.0 meters
→ Pixels per meter: 46.0
```

## Next Steps

After successful calibration:

1. Verify `calibration.json` exists
2. Check that `pixels_per_meter` value is reasonable (30-100 typical)
3. Proceed to detection and tracking setup

---

**Support**: If you encounter issues, check the troubleshooting section or open an issue on GitHub.
