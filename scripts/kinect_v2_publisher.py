#!/usr/bin/env python3
"""
Kinect v2 Headless Gesture Publisher
Production use on Raspberry Pi 4/5 with USB 3.0.
"""

import os
import sys
import time
import signal
import logging
import threading

import numpy as np

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

try:
    from pylibfreenect2 import Freenect2, SyncMultiFrameListener, FrameType
except ImportError:
    print("ERROR: pylibfreenect2 not installed.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("publisher_v2")

mqtt_client = build_mqtt_client("kinect_v2_publisher")
matcher     = GestureMatcher()

fn = Freenect2()
if fn.enumerateDevices() == 0:
    log.error("No Kinect v2 found.")
    sys.exit(1)

serial   = fn.getDeviceSerialNumber(0)
device   = fn.openDevice(serial)
listener = SyncMultiFrameListener(FrameType.Depth)
device.setIrAndDepthFrameListener(listener)
device.start()

gesture_sequence: list[np.ndarray] = []
gesture_arm_side  = "unknown"
stable_count      = 0
last_hand_time    = 0.0
last_gesture_time = 0.0
collecting        = False
running           = True


def shutdown(signum, frame_arg) -> None:
    global running
    running = False
    device.stop()
    device.close()
    try:
        mqtt_client.loop_stop(force=True)
        mqtt_client.disconnect()
    except Exception:
        pass
    log.info("Stopped.")
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


def capture_loop() -> None:
    global gesture_sequence, gesture_arm_side
    global stable_count, last_hand_time, collecting

    while running:
        frames      = listener.waitForNewFrame()
        depth_frame = frames["depth"]
        depth       = depth_frame.asarray().astype(np.uint16).copy()
        listener.release(frames)

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
                        log.info(f"  DTW dist={dist:.3f}  arm={gesture_arm_side}")
                gesture_sequence = []


if __name__ == "__main__":
    log.info("Kinect v2 headless publisher starting")
    log.info("Templates: " +
             "  ".join(f"{k}:{len(v)}" for k, v in matcher.templates.items()))
    log.info("Press Ctrl+C to stop.")
    capture_loop()
