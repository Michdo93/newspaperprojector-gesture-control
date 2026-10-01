# Newspaper Projector Gesture Control

Gesture recognition system for the [Newspaper Projector](https://github.com/Michdo93/newspaperprojector).
Runs on a Raspberry Pi with a Microsoft Kinect camera (v1 or v2), detects hand gestures via DTW template matching on depth frames, and publishes commands to the Newspaper Projector via MQTT.

---

## Supported Hardware

| Component | Kinect v1 | Kinect v2 |
|---|---|---|
| Camera | Xbox 360 Kinect | Xbox One Kinect |
| USB | USB 2.0 | USB 3.0 (mandatory) |
| Raspberry Pi | 3B+ or newer | Pi 4 / Pi 5 only |
| Library | libfreenect | pylibfreenect2 |

---

## Repository Structure

```
newspaperprojector-gesture-control/
├── config/
│   ├── __init__.py
│   └── settings.py.example      # All configuration parameters
├── scripts/
│   ├── kinect_helper.py         # Shared hand detection + visualisation
│   ├── dtw_matcher.py           # DTW template matching engine
│   ├── mqtt_helper.py           # MQTT connection with optional TLS
│   ├── kinect_v1_recorder.py    # Template recorder with web UI (v1)
│   ├── kinect_v1_webviewer.py   # Live depth viewer + publisher (v1)
│   ├── kinect_v1_publisher.py   # Headless publisher for production (v1)
│   ├── kinect_v2_recorder.py    # Template recorder with web UI (v2)
│   ├── kinect_v2_webviewer.py   # Live depth viewer + publisher (v2)
│   └── kinect_v2_publisher.py   # Headless publisher for production (v2)
├── templates/
│   ├── SCROLL_UP/               # .npy template files
│   ├── SCROLL_DOWN/
│   ├── PAGE_NEXT/
│   └── PAGE_PREV/
├── etc/
│   └── systemd/system/
│       └── gesture-control.service
├── requirements.txt
└── README.md
```

---

## Installation

### 1. System packages

```bash
sudo apt-get update && sudo apt-get upgrade -y

# Common dependencies
sudo apt-get install -y \
    python3-pip python3-numpy \
    libusb-1.0-0-dev \
    cmake build-essential

# Kinect v1 only
sudo apt-get install -y libfreenect-dev libfreenect0.5

# Kinect v2 only (requires USB 3.0 — Pi 4/5)
sudo apt-get install -y libfreenect2-dev
```

### 2. Python packages

```bash
# Kinect v1
pip3 install freenect numpy opencv-python flask paho-mqtt \
    --break-system-packages

# Kinect v2 (additional)
pip3 install pylibfreenect2 --break-system-packages
```

### 3. udev rules (Kinect v1)

```bash
sudo nano /etc/udev/rules.d/51-kinect.rules
```

```
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02c2", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02ad", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02ae", MODE="0666", GROUP="plugdev"
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG plugdev pi
```

Reconnect the Kinect USB cable after applying the rules.

### 4. Clone repository

```bash
cd ~
git clone https://github.com/Michdo93/newspaperprojector-gesture-control.git
cd newspaperprojector-gesture-control
```

---

## Configuration

At first you have to copy the `config/settings.py.example` to `config/settings.py`:

```bash
cp config/settings.py.example config/settings.py
```

Then edit `config/settings.py` to match your environment.

### MQTT — local testing (no TLS)

```python
MQTT_BROKER   = "127.0.0.1"
MQTT_PORT     = 1883
MQTT_USERNAME = "projector"
MQTT_PASSWORD = ""
MQTT_CA_CERT  = ""
```

### MQTT — production (TLS, connecting to Newspaper Projector BBB)

Copy the CA certificate from the BeagleBone Black:

```bash
scp debian@newspaperprojector.local:/etc/mosquitto/certs/ca.crt ~/newspaperprojector-gesture-control/certs/newspaperprojector.crt
```

Then update `config/settings.py`:

```python
MQTT_BROKER   = "192.168.x.x"        # IP of the BBB
MQTT_USERNAME = "projector"
MQTT_PASSWORD = "changeme"            # your Mosquitto password
MQTT_CA_CERT  = "/home/pi/newspaperprojector-gesture-control/certs/newspaperprojector.crt"    # path to the copied certificate
```

The `mqtt_helper.py` module automatically selects port 8883 and enables
TLS when `MQTT_CA_CERT` is set to a non-empty string.

### Camera mode

```python
CAMERA_MODE = "CEILING"   # camera mounted top-down
# or
CAMERA_MODE = "DESK"      # camera on a desk, pointing forward

MIRROR_X = True           # set True if PAGE_NEXT/PAGE_PREV appear swapped
```

### Depth zone calibration

The most important parameters. Run the web viewer first and check the
HUD statistics to determine the correct values for your setup.

```python
DEPTH_MIN      = 600    # mm — ignore objects closer (camera housing noise)
DEPTH_MAX      = 2000   # mm — absolute cutoff
DEPTH_MAX_HAND = 870    # mm — background threshold
                         # Objects beyond this are floor/walls/body
                         # Example: camera at 210 cm, floor at ~2100 mm,
                         # hand raised at ~130 cm → depth ~800 mm
                         # Set DEPTH_MAX_HAND to ~870 mm
```

---

## Workflow

### Step 1 — Calibrate depth zone

Run the web viewer and open `http://<pi-ip>:8080` in a browser.

```bash
# Kinect v1
python3 scripts/kinect_v1_webviewer.py

# Kinect v2
python3 scripts/kinect_v2_webviewer.py
```

With an **empty scene** (no person, no hand):
- The image should be **mostly dark grey** (background) and near-black (out of range).
- If large coloured areas appear, `DEPTH_MAX_HAND` is too high — reduce it.

Check the HUD bottom line:
```
Scene — min: 916 mm  mean: 1034 mm  max: 2047 mm  | Hand: 0 mm
```

Set `DEPTH_MAX_HAND` to a value below `min` (e.g. `min - 50 mm`).
In the example above: `DEPTH_MAX_HAND = 860`.

### Step 2 — Record gesture templates

Run the recorder with the desired gesture label:

```bash
# Kinect v1
python3 scripts/kinect_v1_recorder.py --gesture SCROLL_UP

# Kinect v2
python3 scripts/kinect_v2_recorder.py --gesture SCROLL_UP
```

Open `http://<pi-ip>:8080` in a browser.

**Recording procedure:**
1. Select the gesture in the browser (or pass `--gesture` on the command line).
2. Stand in your normal position — the camera should see your hand area.
3. Click **Record** (or press **R**).
4. Perform the gesture motion at a natural speed.
5. Click **Stop & Save** (or press **S**) when done.
6. Repeat 3–5 times per gesture, varying:
   - Left arm vs. right arm
   - Standing slightly left, centre, right
   - Different distances from the camera
7. Repeat for all four gestures: `SCROLL_UP`, `SCROLL_DOWN`, `PAGE_NEXT`, `PAGE_PREV`.

Templates are saved as `.npy` files under `templates/<GESTURE>/`.

**Gesture definitions (CEILING mode — camera pointing down):**

| Gesture | Motion |
|---|---|
| `SCROLL_UP` | Raise hand toward camera, then lower back |
| `SCROLL_DOWN` | Lower hand away from camera, then raise back |
| `PAGE_NEXT` | Swipe hand right, then return |
| `PAGE_PREV` | Swipe hand left, then return |

**Gesture definitions (DESK mode — camera pointing forward):**

| Gesture | Motion |
|---|---|
| `SCROLL_UP` | Move hand upward in frame, then return |
| `SCROLL_DOWN` | Move hand downward in frame, then return |
| `PAGE_NEXT` | Swipe hand right, then return |
| `PAGE_PREV` | Swipe hand left, then return |

### Step 3 — Test with web viewer

After recording, restart the web viewer. It loads templates automatically.

Perform gestures and verify:
- State changes from `WAITING` → `COLLECTING` → back to `WAITING`
- Correct gesture label appears in the top bar after the hand leaves
- MQTT messages are published (check with `mosquitto_sub`)

```bash
# On any machine on the network
mosquitto_sub -h <pi-ip> -t "projector/command/gesture"
```

### Step 4 — Production (headless)

Once templates are recorded and verified:

```bash
# Kinect v1
python3 scripts/kinect_v1_publisher.py

# Kinect v2
python3 scripts/kinect_v2_publisher.py
```

### Step 5 — Run as a systemd service

```bash
sudo cp etc/systemd/system/gesture-control.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gesture-control
sudo systemctl start gesture-control
sudo systemctl status gesture-control
```

For Kinect v2, edit the service file and change `kinect_v1_publisher.py`
to `kinect_v2_publisher.py`.

---

## DTW Template Matching

### How it works

Each recorded template is a sequence of feature vectors:

```
[normalised_centroid_x, normalised_centroid_y, normalised_depth]
```

All values are in [0, 1], making the features resolution-independent.

Before matching, sequences are converted to **delta sequences** (frame-to-frame
differences). This makes matching **translation-invariant** — the user can stand
anywhere in the frame and the gesture is still recognised, because only the
relative motion matters.

DTW (Dynamic Time Warping) compares the delta sequences elastically,
tolerating differences in gesture speed. The normalised DTW distance is
compared against `DTW_MAX_DISTANCE`. If below the threshold, the gesture is
published.

### Arm side detection

The arm side is determined from the centroid x position relative to the
frame centre (`ARM_SPLIT_RATIO = 0.5`). The arm side is logged with each
matched gesture but does not affect which MQTT command is sent — both arms
trigger the same gesture commands.

### Tuning parameters

| Parameter | Effect |
|---|---|
| `DTW_MAX_DISTANCE` | Lower = stricter matching. Increase if gestures are not recognised; decrease if wrong gestures fire. |
| `MIN_GESTURE_FRAMES` | Minimum frames to form a valid gesture. Increase to ignore very short movements. |
| `GESTURE_END_TIMEOUT` | Seconds without hand detection before the sequence is evaluated. |
| `STABLE_FRAMES` | Frames the hand must be continuously visible before collection starts. |

---

## Troubleshooting

**`Can't open device` error:**
- Reconnect the Kinect USB cable.
- Check udev rules: `ls -la /dev/bus/usb/001/` — all Kinect devices should show `crw-rw-rw-`.
- Run as root once to verify: `sudo python3 scripts/kinect_v1_publisher.py`

**All frames are coloured (no grey background):**
- `DEPTH_MAX_HAND` is too high. Run the web viewer with an empty scene,
  check `Scene min` in the HUD, and set `DEPTH_MAX_HAND` below that value.

**Gestures fire in empty room:**
- Reduce `DEPTH_MAX_HAND`.
- Increase `MIN_HAND_PIXELS`.
- Increase `STABLE_FRAMES`.

**Gestures not recognised:**
- Record more templates (3–5 per gesture, varied arm and position).
- Increase `DTW_MAX_DISTANCE` slightly (e.g. from 0.35 to 0.45).
- Check that `MIN_GESTURE_FRAMES` is not too high.

**PAGE_NEXT / PAGE_PREV swapped:**
- Set `MIRROR_X = True` (or `False` if already True).

**Kinect v2 not found:**
- Ensure USB 3.0 connection. Kinect v2 does not work on USB 2.0.
- Only Raspberry Pi 4 and Pi 5 have USB 3.0.

---

## MQTT Topics

| Topic | Payloads | Description |
|---|---|---|
| `projector/command/gesture` | `PAGE_NEXT` `PAGE_PREV` `SCROLL_UP` `SCROLL_DOWN` | Published by this system |

The Newspaper Projector subscribes to this topic and translates the
commands to xdotool keypresses in Chromium.

---

## License

MIT
