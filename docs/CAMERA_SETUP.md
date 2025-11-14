# Camera Setup Guide - TP-Link Tapo C320WS

## 1. Initial Camera Setup

### 1.1 Physical Installation
1. Mount camera at 3-4m height
2. Connect AC power
3. Ensure camera has clear view of road

### 1.2 Tapo App Setup
1. Download **Tapo** app (iOS/Android)
2. Create TP-Link account if needed
3. Add camera to app:
   - Open Tapo app
   - Tap "+" to add device
   - Select "Tapo C320WS"
   - Follow on-screen instructions
4. Connect camera to WiFi network

## 2. Enable RTSP Stream

### 2.1 Camera Settings

1. Open Tapo app
2. Select your C320WS camera
3. Go to **Settings** (gear icon)
4. Go to **Advanced Settings**
5. Find **Camera Account** section
6. Create username and password for RTSP access
   - Example: `admin` / `your_secure_password`
   - **Write these down - you'll need them!**

### 2.2 Enable RTSP

1. In Advanced Settings, find **RTSP**
2. Toggle **RTSP** to **ON**
3. Note the RTSP port (default: 554)

### 2.3 Get Camera IP Address

1. In Tapo app, go to camera settings
2. Go to **Device Info**
3. Note the **IP Address**
   - Example: `192.168.1.100`

**Important**: Set static IP or DHCP reservation for camera to prevent IP changes!

## 3. RTSP URL Format

### 3.1 Standard Format

```
rtsp://username:password@camera_ip:port/stream1
```

### 3.2 Example URLs

**High Quality Stream (stream1)**:
```
rtsp://admin:mypassword@192.168.1.100:554/stream1
```

**Low Quality Stream (stream2)** - for testing or low bandwidth:
```
rtsp://admin:mypassword@192.168.1.100:554/stream2
```

### 3.3 Stream Options

| Stream | Resolution | Bitrate | Use Case |
|--------|-----------|---------|----------|
| stream1 | 2K (2304x1296) | High | Production use |
| stream2 | 640x360 | Low | Testing, low bandwidth |

## 4. Test RTSP Connection

### 4.1 Using FFmpeg (Ubuntu)

Install FFmpeg:
```bash
sudo apt update
sudo apt install ffmpeg
```

Test stream:
```bash
ffplay rtsp://admin:password@192.168.1.100:554/stream1
```

Press 'q' to quit.

### 4.2 Using VLC Player

1. Install VLC:
   ```bash
   sudo apt install vlc
   ```

2. Open VLC
3. Media → Open Network Stream
4. Enter RTSP URL
5. Click Play

### 4.3 Using Python Script

```bash
# Test connection only
python test_camera.py --rtsp-url "rtsp://admin:password@192.168.1.100:554/stream1" --test-only

# Test and display stream
python test_camera.py --rtsp-url "rtsp://admin:password@192.168.1.100:554/stream1"
```

## 5. Configure Project

### 5.1 Update config.yaml

Edit `config.yaml`:

```yaml
camera:
  rtsp_url: "rtsp://admin:your_password@192.168.1.100:554/stream1"
  fps: 15
  resolution: [1920, 1080]
  reconnect_timeout: 30
```

**Security Note**: Don't commit passwords to git! Use environment variables:

```yaml
camera:
  rtsp_url: "${CAMERA_RTSP_URL}"
```

Then create `.env` file:
```bash
CAMERA_RTSP_URL=rtsp://admin:password@192.168.1.100:554/stream1
```

### 5.2 Test Configuration

```bash
python test_camera.py --config config.yaml
```

## 6. Troubleshooting

### 6.1 Cannot Connect

**Check 1: Network connectivity**
```bash
ping 192.168.1.100
```

**Check 2: Port open**
```bash
nc -zv 192.168.1.100 554
```

**Check 3: RTSP enabled**
- Verify in Tapo app that RTSP is ON
- Try rebooting camera

**Check 4: Credentials**
- Verify username and password are correct
- No special characters in URL (URL-encode if needed)

### 6.2 Connection Drops

**Solution 1: Set static IP**
- In router settings, assign static IP to camera MAC address
- Or use DHCP reservation

**Solution 2: Increase reconnect timeout**
```yaml
camera:
  reconnect_timeout: 60  # increase to 60 seconds
```

**Solution 3: Check WiFi signal**
- Camera should have strong WiFi signal
- Consider moving router closer or using WiFi extender

### 6.3 Poor Video Quality

**Solution 1: Use high quality stream**
```yaml
camera:
  rtsp_url: "rtsp://admin:password@192.168.1.100:554/stream1"  # Use stream1, not stream2
```

**Solution 2: Adjust camera settings in Tapo app**
- Video Quality → High
- Night Vision → Color Night Vision

**Solution 3: Check bandwidth**
- Ensure WiFi network has sufficient bandwidth
- Minimize other devices on network during monitoring

### 6.4 High CPU Usage

**Solution 1: Reduce FPS**
```yaml
camera:
  fps: 10  # reduce from 15 to 10
```

**Solution 2: Use lower resolution**
```yaml
camera:
  resolution: [1280, 720]  # reduce from 1920x1080
```

**Solution 3: Use stream2 for testing**
```yaml
camera:
  rtsp_url: "rtsp://admin:password@192.168.1.100:554/stream2"
```

## 7. Camera Settings Recommendations

### 7.1 Tapo App Settings

**Video Settings**:
- Video Quality: **High**
- Resolution: **2K**
- Video Encoding: **H.264**

**Night Vision**:
- Mode: **Color Night Vision** (if sufficient ambient light)
- Or: **Auto** (switches based on light level)

**Detection Settings** (disable to reduce load):
- Motion Detection: **OFF** (we handle this in software)
- Person Detection: **OFF**
- Vehicle Detection: **OFF**
- Recording: **OFF** (we record via RTSP)

**Advanced Settings**:
- RTSP: **ON**
- Camera Account: Set username/password

### 7.2 Optimal Placement

**Height**: 3-4 meters
**Angle**: Perpendicular to road (90 degrees)
**Coverage**: 10-20 meters of road
**Lighting**: Ensure camera can see road clearly day and night

**Avoid**:
- Direct sunlight into lens
- Obstructions (trees, poles)
- Backlighting issues

## 8. Security Best Practices

### 8.1 Network Security

1. **Use strong password** for RTSP account
2. **Isolate camera** on separate VLAN if possible
3. **Disable UPnP** in router (prevent external access)
4. **Use local network only** (no cloud access needed)

### 8.2 Credentials Management

**Never commit credentials to git!**

Use environment variables:
```bash
# .env file (gitignored)
CAMERA_USERNAME=admin
CAMERA_PASSWORD=your_secure_password
CAMERA_IP=192.168.1.100
CAMERA_PORT=554
```

Load in code:
```python
from dotenv import load_dotenv
import os

load_dotenv()

rtsp_url = f"rtsp://{os.getenv('CAMERA_USERNAME')}:{os.getenv('CAMERA_PASSWORD')}@{os.getenv('CAMERA_IP')}:{os.getenv('CAMERA_PORT')}/stream1"
```

## 9. Performance Optimization

### 9.1 For Real-time Monitoring

```yaml
camera:
  fps: 15
  resolution: [1920, 1080]
  buffer_size: 1  # minimal latency
```

### 9.2 For Batch Processing

```yaml
camera:
  fps: 10
  resolution: [1280, 720]
  buffer_size: 3  # smoother playback
```

### 9.3 For Low-end Hardware

```yaml
camera:
  fps: 10
  resolution: [640, 480]
  buffer_size: 1
```

## 10. Next Steps

After camera setup is complete:

1. **Calibrate camera** for speed measurement
   ```bash
   python calibration/calibrate.py
   ```

2. **Test detection pipeline**
   ```bash
   python main.py --test
   ```

3. **Start monitoring**
   ```bash
   python main.py --config config.yaml
   ```

---

**Support**:
- TP-Link Support: https://www.tp-link.com/support/
- Tapo C320WS Manual: https://www.tp-link.com/en/support/download/tapo-c320ws/

**Common RTSP Issues**:
- https://community.tp-link.com/en/home/forum/topic/234563
- https://www.tp-link.com/en/support/faq/2680/
