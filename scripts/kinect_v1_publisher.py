#!/usr/bin/env python3
"""
Kinect v1 Headless Gesture Publisher
Runs without a display — intended for production use on the Raspberry Pi.
Loads templates, collects depth sequences, matches via DTW, publishes MQTT.
"""

import os
import sys
import time
import signal
import logging

import freenect
import numpy as np

# Suppress libfreenect noise
_devnull = open(os.devnull, "w")
os.dup2(_devnull.fileno(), sys.stderr.fileno())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    STABLE_FRAMES, GESTURE_END_TIMEOUT,
    MIN_GESTURE_FRAMES, COOLDOWN, MQTT_TOPIC
)
from scripts.kinect_helper import detect_hand
from scripts.dtw_matcher import GestureMatcher
from scripts.mqtt_helper import build_mqtt_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout
)
log = logging.getLogger("publisher")

# ── MQTT + matcher ────────────────────────────────────────────────────────────
mqtt_client = build_mqtt_client("kinect_v1_publisher")
matcher     = GestureMatcher()

# ── State ─────────────────────────────────────────────────────────────────────
gesture_sequence: list[np.ndarray] = []
gesture_arm_side  = "unknown"
stable_count      = 0
last_hand_time    = 0.0
last_gesture_time = 0.0
collecting        = False
depth_stream_started = False
running           = True


def shutdown(signum, frame_arg) -> None:
    global running
    running = False
    log.info("Stopped by user.")
    try:
        mqtt_client.loop_stop(force=True)
        mqtt_client.disconnect()
    except Exception:
        pass
    sys.exit(0)


signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)


def publish_gesture(gesture: str) -> None:
    global last_gesture_time
    now = time.time()
    if now - last_gesture_time < COOLDOWN:
        return
    last_gesture_time = now
    mqtt_client.publish(MQTT_TOPIC, gesture)
    log.info(f"Published: {gesture}")


def process_depth_frame(dev, depth, timestamp) -> None:
    global gesture_sequence, gesture_arm_side
    global stable_count, last_hand_time, collecting

    if not running:
        return

    detection = detect_hand(depth)
    now       = time.time()

    if detection is not None:
        stable_count  += 1
        last_hand_time = now

        if stable_count >= STABLE_FRAMES:
            if not collecting:
                collecting       = True
                gesture_sequence = []
                gesture_arm_side = detection.arm_side
                log.debug("Collection started.")
            gesture_sequence.append(detection.as_feature_vector())
            gesture_arm_side = detection.arm_side
    else:
        stable_count = 0

        if collecting and now - last_hand_time > GESTURE_END_TIMEOUT:
            collecting = False
            if len(gesture_sequence) >= MIN_GESTURE_FRAMES:
                label, dist = matcher.match(gesture_sequence, gesture_arm_side)
                if label:
                    publish_gesture(label)
                    log.info(f"  DTW dist={dist:.3f}  arm={gesture_arm_side}  "
                             f"frames={len(gesture_sequence)}")
                else:
                    log.debug(f"No template match (best dist={dist:.3f})")
            else:
                log.debug(f"Sequence too short ({len(gesture_sequence)} frames) — ignored")
            gesture_sequence = []


def setup_device(dev, ctx) -> None:
    global depth_stream_started
    if not depth_stream_started:
        freenect.set_depth_mode(dev, freenect.RESOLUTION_MEDIUM, freenect.DEPTH_11BIT)
        freenect.set_depth_callback(dev, process_depth_frame)
        freenect.start_depth(dev)
        depth_stream_started = True
        log.info("Depth stream active.")


if __name__ == "__main__":
    log.info(f"Headless publisher starting")
    log.info(f"Templates: " +
             "  ".join(f"{k}:{len(v)}" for k, v in matcher.templates.items()))
    log.info("Press Ctrl+C to stop.")
    freenect.runloop(depth=process_depth_frame, body=setup_device)
