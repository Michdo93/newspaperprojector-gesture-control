#!/usr/bin/env python3
"""
Kinect v1 (Xbox 360) Gesture Control
Camera faces top-down
Runs on Raspberry Pi 3B+
Sends gestures via MQTT to BeagleBone Black
"""

import freenect
import numpy as np
import paho.mqtt.client as mqtt
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
log = logging.getLogger("kinect_v1")

# ── Configuration ─────────────────────────────────────────────
BROKER  = "192.168.0.5"   # Set IP of BBB or MQTT broker
PORT    = 1883
TOPIC   = "projector/command/gesture"

# Minimum motion in pixels to trigger a gesture
SWIPE_THRESHOLD   = 40    # px for left/right/up/down
DEPTH_THRESHOLD   = 80    # mm change for up/down (top-down camera)

# Cooldown between two gestures (seconds)
COOLDOWN = 1.2

# Depth range in which the hand is detected (mm)
DEPTH_MIN = 400   # ignore closer than 40cm
DEPTH_MAX = 1500  # ignore further than 150cm
# ──────────────────────────────────────────────────────────────

mqtt_client = mqtt.Client()
mqtt_client.connect(BROKER, PORT, 60)
mqtt_client.loop_start()

last_gesture_time = 0
last_hand_x       = None
last_hand_y       = None
last_hand_depth   = None  # Average depth of the hand

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
    Camera points downward → Hand raised = smaller depth value.
    Finds the closest object within the defined depth range.
    Returns (y, x, mean_depth) or None.
    """
    img = depth_image.astype(np.float32)

    # Only objects within a valid depth range
    mask = (img > DEPTH_MIN) & (img < DEPTH_MAX)
    if not np.any(mask):
        return None

    # Find the closest point (smallest depth value = closest hand)
    filtered_img = np.where(mask, img, 9999)
    min_pos = np.unravel_index(np.argmin(filtered_img), img.shape)

    # Region around the closest point for more stable measurement
    y, x = min_pos
    h, w = img.shape
    y0, y1 = max(0, y - 20), min(h, y + 20)
    x0, x1 = max(0, x - 20), min(w, x + 20)
    region = img[y0:y1, x0:x1]
    region_mask = (region > DEPTH_MIN) & (region < DEPTH_MAX)

    if not np.any(region_mask):
        return None

    mean_depth = float(np.mean(region[region_mask]))
    return (int(y), int(x), mean_depth)

def process_depth_image(depth, _timestamp):
    global last_hand_x, last_hand_y, last_hand_depth

    result = find_hand(depth)

    if result is None:
        # Hand not visible — reset state
        last_hand_x     = None
        last_hand_y     = None
        last_hand_depth = None
        return

    hand_y, hand_x, hand_depth = result

    if last_hand_x is None:
        # Initialize first frame
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

    # Camera points DOWNWARD:
    # Hand raises  → depth value SMALLER (closer to camera) → UP
    # Hand lowers → depth value LARGER  (further from camera) → DOWN

    if abs_depth > DEPTH_THRESHOLD and abs_depth > abs_x and abs_depth > abs_y:
        # Vertical gesture (up/down) dominates
        if delta_depth < -DEPTH_THRESHOLD:
            send_gesture("SCROLL_UP")      # Hand raised
        elif delta_depth > DEPTH_THRESHOLD:
            send_gesture("SCROLL_DOWN")    # Hand lowered

    elif abs_x > SWIPE_THRESHOLD and abs_x > abs_y:
        # Horizontal gesture (left/right) dominates
        if delta_x > SWIPE_THRESHOLD:
            send_gesture("PAGE_NEXT")   # Hand to the right
        elif delta_x < -SWIPE_THRESHOLD:
            send_gesture("PAGE_PREV")    # Hand to the left

    elif abs_y > SWIPE_THRESHOLD and abs_y > abs_x:
        # Depth axis direction (front/back from camera perspective)
        # Optional: use for scrolling
        pass

    # Save current position
    last_hand_x     = hand_x
    last_hand_y     = hand_y
    last_hand_depth = hand_depth

log.info(f"Kinect v1 Publisher starting — Broker: {BROKER}:{PORT}")
log.info("Camera orientation: top-down")
log.info("Gestures: Raise hand=Up, Lower hand=Down, Left/Right=Item navigation")
log.info("Ctrl+C to terminate")

try:
    freenect.runloop(depth=process_depth_image)
except KeyboardInterrupt:
    log.info("Terminated.")
finally:
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
