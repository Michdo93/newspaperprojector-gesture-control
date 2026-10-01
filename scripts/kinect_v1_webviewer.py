#!/usr/bin/env python3
"""
Kinect v1 Web Viewer + DTW Gesture Publisher
Live depth stream with gesture recognition overlay.
Publishes matched gestures via MQTT.

Open in browser: http://<pi-ip>:8080
"""

import os
import sys
import time
import signal
import logging
import threading

import freenect
import numpy as np
import cv2
from flask import Flask, Response, render_template_string

_devnull = open(os.devnull, "w")
os.dup2(_devnull.fileno(), sys.stderr.fileno())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    JPEG_QUALITY, STREAM_FPS, WEB_PORT,
    DEPTH_MIN, DEPTH_MAX_HAND,
    STABLE_FRAMES, GESTURE_END_TIMEOUT,
    MIN_GESTURE_FRAMES, COOLDOWN, MQTT_TOPIC
)
from scripts.kinect_helper import detect_hand, build_heatmap
from scripts.dtw_matcher import GestureMatcher
from scripts.mqtt_helper import build_mqtt_client

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("webviewer_v1")

mqtt_client = build_mqtt_client("kinect_v1_webviewer")
matcher     = GestureMatcher()

frame_lock        = threading.Lock()
current_vis_frame = None

gesture_sequence: list[np.ndarray] = []
gesture_arm_side  = "unknown"
stable_count      = 0
last_hand_time    = 0.0
last_gesture_time = 0.0
last_gesture_text = "—"
collecting        = False
depth_stream_started = False
running           = True

depth_stats   = {"min": 0, "mean": 0, "max": 0, "hand": 0}
frame_counter = 0


def shutdown(signum, frame_arg) -> None:
    global running
    running = False
    log.info("Stopped.")
    try:
        mqtt_client.loop_stop(force=True)
        mqtt_client.disconnect()
    except Exception:
        pass
    sys.exit(0)


signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)


def publish_gesture(gesture: str) -> None:
    global last_gesture_time, last_gesture_text
    now = time.time()
    if now - last_gesture_time < COOLDOWN:
        return
    last_gesture_time = now
    last_gesture_text = gesture
    mqtt_client.publish(MQTT_TOPIC, gesture)
    log.info(f"Published: {gesture}")


def process_depth_frame(dev, depth, timestamp) -> None:
    global gesture_sequence, gesture_arm_side
    global stable_count, last_hand_time, collecting
    global current_vis_frame, frame_counter

    if not running:
        return

    frame_counter += 1
    if frame_counter % 30 == 0:
        img   = depth.astype(np.float32)
        valid = img[(img > 100) & (img < 4000)]
        if len(valid) > 0:
            depth_stats.update(min=int(valid.min()),
                               mean=int(valid.mean()),
                               max=int(valid.max()))

    detection = detect_hand(depth)
    now       = time.time()

    if detection is not None:
        stable_count  += 1
        last_hand_time = now
        depth_stats["hand"] = int(detection.mean_depth)

        if stable_count >= STABLE_FRAMES:
            if not collecting:
                collecting       = True
                gesture_sequence = []
                gesture_arm_side = detection.arm_side
            gesture_sequence.append(detection.position())
            gesture_arm_side = detection.arm_side
    else:
        stable_count    = 0
        depth_stats["hand"] = 0

        if collecting and now - last_hand_time > GESTURE_END_TIMEOUT:
            collecting = False
            if len(gesture_sequence) >= MIN_GESTURE_FRAMES:
                label, dist = matcher.match(gesture_sequence, gesture_arm_side)
                if label:
                    publish_gesture(label)
                    log.info(f"  DTW dist={dist:.4f}  arm={gesture_arm_side}")
                else:
                    log.debug(f"No match (best dist={dist:.4f})")
            gesture_sequence = []

    # ── Visualisation ──────────────────────────────────────────────────────
    vis  = build_heatmap(depth)
    h, w = vis.shape[:2]

    if detection is not None:
        hy, hx = detection.y, detection.x
        y0, y1 = max(0, hy - 25), min(h, hy + 25)
        x0, x1 = max(0, hx - 25), min(w, hx + 25)
        col = (0, 80, 255) if collecting else (0, 255, 0)
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
        cv2.drawMarker(vis, (hx, hy), (0, 255, 255),
                       cv2.MARKER_CROSS, 24, 2)
        cv2.putText(vis,
                    f"{int(detection.mean_depth)} mm  {detection.arm_side}",
                    (hx + 14, hy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    state_text   = f"COLLECTING ({len(gesture_sequence)} fr)" \
                   if collecting else "WAITING"
    state_colour = (0, 150, 255) if collecting else (100, 100, 100)
    cv2.rectangle(vis, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(vis, state_text, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_colour, 2)
    g_colour = (0, 255, 80) if last_gesture_text != "—" else (100, 100, 100)
    cv2.putText(vis, f"Gesture: {last_gesture_text}",
                (w // 2 - 80, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, g_colour, 2)

    hud = [
        f"Hand zone: {DEPTH_MIN}–{DEPTH_MAX_HAND} mm  "
        f"| min:{depth_stats['min']} mean:{depth_stats['mean']} "
        f"max:{depth_stats['max']} hand:{depth_stats['hand']} mm",
        "Templates: " +
        "  ".join(f"{k}:{len(v)}" for k, v in matcher.templates.items()),
    ]
    bar_h = len(hud) * 24 + 8
    cv2.rectangle(vis, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    for i, line in enumerate(hud):
        cv2.putText(vis, line, (8, h - bar_h + 20 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    vis = cv2.resize(vis, (960, 720), interpolation=cv2.INTER_NEAREST)
    ret, jpeg = cv2.imencode(".jpg", vis,
                             [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if ret:
        with frame_lock:
            current_vis_frame = jpeg.tobytes()


def setup_device(dev, ctx) -> None:
    global depth_stream_started
    if not depth_stream_started:
        freenect.set_depth_mode(dev, freenect.RESOLUTION_MEDIUM, freenect.DEPTH_11BIT)
        freenect.set_depth_callback(dev, process_depth_frame)
        freenect.start_depth(dev)
        depth_stream_started = True
        log.info("Depth stream active.")


app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Kinect v1 Viewer</title>
<style>
  body{background:#0d1117;color:#c9d1d9;font-family:monospace;margin:0;
       display:flex;flex-direction:column;align-items:center;}
  h1{color:#58a6ff;margin:12px 0 4px;font-size:18px;}
  p{color:#8b949e;font-size:12px;margin:2px 0 6px;text-align:center;}
  img{border:2px solid #30363d;border-radius:6px;max-width:100%;}
</style></head>
<body>
  <h1>Kinect v1 — Depth Viewer (DTW matcher)</h1>
  <p>Blue = collecting  |  Green = hand detected  |  Gesture shown top-right</p>
  <img src="/stream" alt="stream">
</body></html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


def generate_frames():
    while True:
        with frame_lock:
            frame = current_vis_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(1.0 / STREAM_FPS)


@app.route("/stream")
def stream():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/reload_templates", methods=["POST"])
def reload_templates():
    matcher.reload()
    return ("", 204)


if __name__ == "__main__":
    log.info(f"Web viewer: http://0.0.0.0:{WEB_PORT}")
    threading.Thread(
        target=lambda: freenect.runloop(
            depth=process_depth_frame, body=setup_device),
        daemon=True
    ).start()
    app.run(host="0.0.0.0", port=WEB_PORT, threaded=True)
