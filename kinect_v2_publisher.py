#!/usr/bin/env python3
"""
Kinect v2 (Xbox One) Gesture Control
Camera faces top-down
Runs on Raspberry Pi 3B+ (USB 3.0 required — Pi 3B+ only has USB 2.0!)
WARNING: Kinect v2 requires USB 3.0 — Pi 3B+ only has USB 2.0.
         Recommendation: Use Kinect v1 for Pi 3B+!
         Use this script only for Pi 4 or Pi 5 with USB 3.0.
Sends gestures via MQTT to BeagleBone Black
"""

import paho.mqtt.client as mqtt
import numpy as np
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
log = logging.getLogger("kinect_v2")

# ── Configuration ─────────────────────────────────────────────
BROKER  = "192.168.0.5"   # Set IP of BBB or MQTT broker
PORT    = 1883
TOPIC   = "projector/command/gesture"

SWIPE_THRESHOLD   = 50    # px for left/right
DEPTH_THRESHOLD   = 100   # mm change for up/down
COOLDOWN          = 1.2   # Seconds between gestures

DEPTH_MIN = 500            # mm
DEPTH_MAX = 2000           # mm
# ──────────────────────────────────────────────────────────────

try:
    from pylibfreenect2 import Freenect2, SyncMultiFrameListener
    from pylibfreenect2 import FrameType, Registration, Frame
    import pylibfreenect2 as fn2
except ImportError:
    log.error("pylibfreenect2 is not installed!")
    log.error("sudo pip3 install pylibfreenect2 --break-system-packages")
    exit(1)

mqtt_client = mqtt.Client()
mqtt_client.connect(BROKER, PORT, 60)
mqtt_client.loop_start()

last_gesture_time = 0
last_hand_x       = None
last_hand_y       = None
last_hand_depth   = None

def send_gesture(gesture: str):
    global last_gesture_time
    now = time.time()
    if now - last_gesture_time < COOLDOWN:
        return
    last_gesture_time = now
    mqtt_client.publish(TOPIC, gesture)
    log.info(f"Gesture sent: {gesture}")

def find_hand(depth_image: np.ndarray):
    """
    Finds the closest hand in the depth image.
    Returns (y, x, mean_depth) or None.
    """
    img = depth_image.astype(np.float32)
    mask = (img > DEPTH_MIN) & (img < DEPTH_MAX)
    if not np.any(mask):
        return None

    filtered_img = np.where(mask, img, 99999)
    min_pos = np.unravel_index(np.argmin(filtered_img), img.shape)

    y, x = min_pos
    h, w = img.shape
    y0, y1 = max(0, y - 25), min(h, y + 25)
    x0, x1 = max(0, x - 25), min(w, x + 25)
    region = img[y0:y1, x0:x1]
    region_mask = (region > DEPTH_MIN) & (region < DEPTH_MAX)

    if not np.any(region_mask):
        return None

    mean_depth = float(np.mean(region[region_mask]))
    return (int(y), int(x), mean_depth)

def process_frame(depth_image: np.ndarray):
    global last_hand_x, last_hand_y, last_hand_depth

    result = find_hand(depth_image)

    if result is None:
        last_hand_x     = None
        last_hand_y     = None
        last_hand_depth = None
        return

    hand_y, hand_x, hand_depth = result

    if last_hand_x is None:
        last_hand_x     = hand_x
        last_hand_y     = hand_y
        last_hand_depth = hand_depth
        return

    delta_x     = hand_x     - last_hand_x
    delta_y     = hand_y     - last_hand_y
    delta_depth = hand_depth - last_hand_depth

    abs_x     = abs(delta_x)
    abs_y     = abs(delta_y)
    abs_depth = abs(delta_depth)

    if abs_depth > DEPTH_THRESHOLD and abs_depth > abs_x and abs_depth > abs_y:
        if delta_depth < -DEPTH_THRESHOLD:
            send_gesture("SCROLL_UP")
        elif delta_depth > DEPTH_THRESHOLD:
            send_gesture("SCROLL_DOWN")
    elif abs_x > SWIPE_THRESHOLD and abs_x > abs_y:
        if delta_x > SWIPE_THRESHOLD:
            send_gesture("PAGE_NEXT")
        elif delta_x < -SWIPE_THRESHOLD:
            send_gesture("PAGE_PREV")

    last_hand_x     = hand_x
    last_hand_y     = hand_y
    last_hand_depth = hand_depth

# ── Initialize Kinect v2 ───────────────────────────────────────
log.info("Kinect v2 Publisher starting")
log.warning("WARNING: Pi 3B+ only has USB 2.0 — Kinect v2 requires USB 3.0!")
log.warning("Recommendation: Use Kinect v1 for Pi 3B+")

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    log.error("No Kinect v2 found!")
    exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial)

listener = SyncMultiFrameListener(FrameType.Depth)
device.setIrAndDepthFrameListener(listener)
device.start()

log.info(f"Kinect v2 connected — Broker: {BROKER}:{PORT}")
log.info("Ctrl+C to terminate")

try:
    while True:
        frames = listener.waitForNewFrame()
        depth_frame = frames["depth"]
        depth_image = depth_frame.asarray()
        process_frame(depth_image)
        listener.release(frames)

except KeyboardInterrupt:
    log.info("Terminated.")
finally:
    device.stop()
    device.close()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
