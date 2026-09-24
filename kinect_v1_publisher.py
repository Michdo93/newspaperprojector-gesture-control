#!/usr/bin/env python3
"""
Kinect v1 (Xbox 360) Gesture Publisher
Camera mounted top-down, pointing at the floor.
Detects hand gestures and publishes MQTT commands to the Newspaper Projector.

Gestures:
  Swipe right  → PAGE_NEXT
  Swipe left   → PAGE_PREV
  Raise hand   → SCROLL_UP
  Lower hand   → SCROLL_DOWN
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

# ── Configuration ──────────────────────────────────────────────────────────────
BROKER = "192.168.0.5"          # IP address of the MQTT broker
PORT   = 1883
TOPIC  = "projector/command/gesture"

# Minimum pixel displacement to trigger a horizontal swipe gesture
SWIPE_THRESHOLD = 40            # pixels

# Minimum depth change (mm) to trigger a vertical gesture
DEPTH_THRESHOLD = 80            # millimetres

# Minimum seconds between two gestures (prevents rapid repeat firing)
COOLDOWN = 1.2                  # seconds

# Only detect objects within this depth range (mm from camera)
DEPTH_MIN = 400                 # 40 cm  — ignore objects too close
DEPTH_MAX = 1500                # 150 cm — ignore objects too far

# Minimum connected pixels required to consider a detection valid
# Filters out floor reflections and sensor noise
MIN_HAND_PIXELS = 50
# ──────────────────────────────────────────────────────────────────────────────

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.connect(BROKER, PORT, 60)
mqtt_client.loop_start()

last_gesture_time = 0.0
last_hand_x       = None
last_hand_y       = None
last_hand_depth   = None


def publish_gesture(gesture: str) -> None:
    """Publish a gesture command via MQTT, subject to cooldown."""
    global last_gesture_time
    now = time.time()
    if now - last_gesture_time < COOLDOWN:
        return
    last_gesture_time = now
    mqtt_client.publish(TOPIC, gesture)
    log.info(f"Published: {gesture}")


def detect_hand(depth_frame: np.ndarray):
    """
    Locate the closest hand-sized object in the depth frame.

    With the camera mounted top-down:
      - A raised hand produces a SMALLER depth value (closer to camera).
      - A lowered hand produces a LARGER depth value (further from camera).

    Returns (y, x, mean_depth) if a valid hand region is found, else None.
    """
    img = depth_frame.astype(np.float32)

    # Isolate pixels within the valid depth range
    valid_mask = (img > DEPTH_MIN) & (img < DEPTH_MAX)
    if not np.any(valid_mask):
        return None

    # Reject detections that are too small (noise, reflections)
    if np.sum(valid_mask) < MIN_HAND_PIXELS:
        return None

    # Find the pixel with the smallest depth value (closest point)
    search_img = np.where(valid_mask, img, 9999.0)
    min_pos    = np.unravel_index(np.argmin(search_img), img.shape)
    y, x       = min_pos

    # Sample a small region around the closest point for a stable depth estimate
    h, w = img.shape
    y0, y1 = max(0, y - 20), min(h, y + 20)
    x0, x1 = max(0, x - 20), min(w, x + 20)
    region      = img[y0:y1, x0:x1]
    region_mask = (region > DEPTH_MIN) & (region < DEPTH_MAX)

    if not np.any(region_mask):
        return None

    if np.sum(region_mask) < MIN_HAND_PIXELS:
        return None

    mean_depth = float(np.mean(region[region_mask]))
    return (int(y), int(x), mean_depth)


def process_depth_frame(dev, depth, timestamp) -> None:
    """
    Freenect depth callback.
    Called for every incoming depth frame from the Kinect sensor.
    """
    global last_hand_x, last_hand_y, last_hand_depth

    detection = detect_hand(depth)

    if detection is None:
        # No hand detected — clear stored position, publish nothing
        last_hand_x     = None
        last_hand_y     = None
        last_hand_depth = None
        return

    hand_y, hand_x, hand_depth = detection

    if last_hand_x is None:
        # First frame with a valid detection — store position, do not gesture yet
        last_hand_x     = hand_x
        last_hand_y     = hand_y
        last_hand_depth = hand_depth
        return

    # Calculate displacement since the last frame
    delta_x     = hand_x     - last_hand_x
    delta_y     = hand_y     - last_hand_y
    delta_depth = hand_depth - last_hand_depth

    abs_x     = abs(delta_x)
    abs_y     = abs(delta_y)
    abs_depth = abs(delta_depth)

    # Determine the dominant axis and fire the appropriate gesture
    if abs_depth > DEPTH_THRESHOLD and abs_depth > abs_x and abs_depth > abs_y:
        # Vertical axis (depth) dominates
        if delta_depth < -DEPTH_THRESHOLD:
            publish_gesture("SCROLL_UP")    # Hand moved toward camera (raised)
        elif delta_depth > DEPTH_THRESHOLD:
            publish_gesture("SCROLL_DOWN")  # Hand moved away from camera (lowered)

    elif abs_x > SWIPE_THRESHOLD and abs_x > abs_y:
        # Horizontal axis dominates
        if delta_x > SWIPE_THRESHOLD:
            publish_gesture("PAGE_NEXT")    # Hand swiped right
        elif delta_x < -SWIPE_THRESHOLD:
            publish_gesture("PAGE_PREV")    # Hand swiped left

    # Update stored position
    last_hand_x     = hand_x
    last_hand_y     = hand_y
    last_hand_depth = hand_depth


# ── Entry point ────────────────────────────────────────────────────────────────
log.info(f"Kinect v1 Publisher starting — Broker: {BROKER}:{PORT}, Topic: {TOPIC}")
log.info("Camera orientation: top-down")
log.info("Swipe right=PAGE_NEXT | Swipe left=PAGE_PREV | Raise=SCROLL_UP | Lower=SCROLL_DOWN")
log.info("Press Ctrl+C to stop")

try:
    freenect.runloop(depth=process_depth_frame)
except KeyboardInterrupt:
    log.info("Stopped by user.")
except Exception as e:
    log.error(f"Unexpected error: {e}")
finally:
    try:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    except Exception:
        pass
