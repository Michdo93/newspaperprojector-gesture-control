#!/usr/bin/env python3
"""
Kinect v2 Gesture Recorder — Web UI
Records depth-frame sequences as gesture templates.
Requires: Raspberry Pi 4/5 with USB 3.0, pylibfreenect2

Usage:
  python3 scripts/kinect_v2_recorder.py --gesture SCROLL_UP
"""

import os
import sys
import time
import signal
import logging
import argparse
import threading
import datetime

import numpy as np
import cv2
from flask import Flask, Response, render_template_string, jsonify, request

# Suppress stderr noise
_devnull = open(os.devnull, "w")
os.dup2(_devnull.fileno(), sys.stderr.fileno())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    TEMPLATE_DIR, MIN_GESTURE_FRAMES, MAX_GESTURE_FRAMES,
    GESTURE_END_TIMEOUT, JPEG_QUALITY, STREAM_FPS, WEB_PORT,
    DEPTH_MIN, DEPTH_MAX, DEPTH_MAX_HAND, STABLE_FRAMES
)
from scripts.kinect_helper import detect_hand, build_heatmap

try:
    import pylibfreenect2 as fk2
    from pylibfreenect2 import Freenect2, SyncMultiFrameListener, FrameType
except ImportError:
    print("ERROR: pylibfreenect2 not installed.")
    print("  pip3 install pylibfreenect2 --break-system-packages")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout
)
log = logging.getLogger("recorder_v2")

GESTURE_LABELS = ["SCROLL_UP", "SCROLL_DOWN", "PAGE_NEXT", "PAGE_PREV"]

# ── Kinect v2 init ────────────────────────────────────────────────────────────
fn       = Freenect2()
num_devs = fn.enumerateDevices()
if num_devs == 0:
    log.error("No Kinect v2 found. Check USB 3.0 connection.")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial)
listener = SyncMultiFrameListener(FrameType.Depth)
device.setIrAndDepthFrameListener(listener)
device.start()
log.info("Kinect v2 connected.")

# ── Shared state ──────────────────────────────────────────────────────────────
frame_lock        = threading.Lock()
current_vis_frame = None

recording         = False
record_sequence: list[np.ndarray] = []
record_arm_side   = "unknown"
last_hand_time    = 0.0
stable_count      = 0
running           = True

target_gesture    = "SCROLL_UP"
saved_count       = 0
last_save_path    = ""
depth_stats       = {"min": 0, "mean": 0, "max": 0, "hand": 0}
frame_counter     = 0


def shutdown(signum, frame_arg) -> None:
    global running
    running = False
    device.stop()
    device.close()
    log.info("Stopped.")
    sys.exit(0)


signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)


def save_template(sequence: list[np.ndarray], arm_side: str) -> str:
    folder = os.path.join(TEMPLATE_DIR, target_gesture)
    os.makedirs(folder, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"{ts}_{arm_side}.npy")
    np.save(path, np.stack(sequence))
    log.info(f"Saved: {path}  ({len(sequence)} frames, arm={arm_side})")
    return path


def finalize_recording() -> None:
    global recording, record_sequence, record_arm_side, saved_count, last_save_path
    if len(record_sequence) >= MIN_GESTURE_FRAMES:
        last_save_path = save_template(record_sequence, record_arm_side)
        saved_count   += 1
    else:
        log.warning(f"Too short ({len(record_sequence)} fr) — discarded")
    recording       = False
    record_sequence = []
    record_arm_side = "unknown"


def capture_loop() -> None:
    global recording, record_sequence, record_arm_side
    global last_hand_time, stable_count, frame_counter
    global current_vis_frame

    while running:
        frames = listener.waitForNewFrame()
        depth_frame = frames["depth"]
        depth = depth_frame.asarray().copy()
        listener.release(frames)

        # v2 depth is in mm directly
        depth = depth.astype(np.uint16)

        frame_counter += 1
        if frame_counter % 30 == 0:
            img   = depth.astype(np.float32)
            valid = img[(img > 100) & (img < 4000)]
            if len(valid) > 0:
                depth_stats.update(min=int(valid.min()),
                                   mean=int(valid.mean()),
                                   max=int(valid.max()))

        detection = detect_hand(depth)
        now = time.time()

        if detection is not None:
            stable_count  += 1
            last_hand_time = now
            depth_stats["hand"] = int(detection.mean_depth)

            if recording and stable_count >= STABLE_FRAMES:
                record_sequence.append(detection.as_feature_vector())
                record_arm_side = detection.arm_side
                if len(record_sequence) >= MAX_GESTURE_FRAMES:
                    finalize_recording()
        else:
            stable_count    = 0
            depth_stats["hand"] = 0
            if recording and now - last_hand_time > GESTURE_END_TIMEOUT:
                finalize_recording()

        # Visualise
        vis  = build_heatmap(depth)
        h, w = vis.shape[:2]

        if detection is not None:
            hy, hx = detection.y, detection.x
            y0, y1 = max(0, hy - 25), min(h, hy + 25)
            x0, x1 = max(0, hx - 25), min(w, hx + 25)
            col = (0, 0, 255) if recording else (0, 255, 0)
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
            cv2.drawMarker(vis, (hx, hy), (0, 255, 255),
                           cv2.MARKER_CROSS, 24, 2)
            cv2.putText(vis,
                        f"{int(detection.mean_depth)} mm  {detection.arm_side}",
                        (hx + 14, hy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        rec_text   = f"REC {len(record_sequence)}/{MAX_GESTURE_FRAMES}" \
                     if recording else "READY"
        rec_colour = (0, 0, 255) if recording else (0, 200, 100)
        cv2.rectangle(vis, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(vis, f"Gesture: {target_gesture}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 200, 255), 2)
        cv2.putText(vis, rec_text,
                    (w - 220, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, rec_colour, 2)

        hud = [
            f"Hand zone: {DEPTH_MIN}–{DEPTH_MAX_HAND} mm",
            f"Scene min:{depth_stats['min']} mean:{depth_stats['mean']} "
            f"max:{depth_stats['max']} hand:{depth_stats['hand']} mm",
            f"Saved: {saved_count}  Last: {os.path.basename(last_save_path)}",
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


# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Gesture Recorder v2 — {{ gesture }}</title>
  <style>
    body { background:#0d1117; color:#c9d1d9; font-family:monospace;
           margin:0; display:flex; flex-direction:column; align-items:center; }
    h1   { color:#58a6ff; margin:12px 0 4px; font-size:20px; }
    p    { color:#8b949e; font-size:12px; margin:2px 0; text-align:center; }
    img  { border:2px solid #30363d; border-radius:6px; max-width:100%; margin:8px 0; }
    .btn { padding:12px 32px; font-size:16px; border-radius:8px; cursor:pointer;
           border:none; margin:6px; font-family:monospace; }
    .record { background:#8e1519; color:#fff; }
    .record.active { background:#f85149; }
    .stop   { background:#238636; color:#fff; }
    #status { font-size:14px; color:#e3b341; margin:4px 0; min-height:20px; }
    .gestures { display:flex; gap:8px; flex-wrap:wrap; justify-content:center;
                margin:8px 0; }
    .gsel { padding:8px 16px; border-radius:6px; cursor:pointer;
            border:2px solid #30363d; background:#21262d;
            color:#c9d1d9; font-family:monospace; font-size:13px; }
    .gsel.selected { border-color:#58a6ff; color:#58a6ff; }
  </style>
</head>
<body>
  <h1>Gesture Recorder — Kinect v2</h1>
  <p>Select a gesture, click Record, perform the motion, click Stop &amp; Save.</p>
  <div class="gestures">
    {% for g in gestures %}
    <button class="gsel {% if g == gesture %}selected{% endif %}"
            onclick="selectGesture('{{ g }}')">{{ g }}</button>
    {% endfor %}
  </div>
  <img src="/stream" alt="depth stream">
  <div>
    <button class="btn record" id="recBtn" onclick="startRecord()">⏺ Record</button>
    <button class="btn stop"   onclick="stopRecord()">⏹ Stop &amp; Save</button>
  </div>
  <div id="status"></div>
  <p id="saved">Saved: 0 templates</p>
  <script>
    function selectGesture(g) {
      fetch("/set_gesture", {method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({gesture: g})
      }).then(r=>r.json()).then(d=>{
        document.querySelectorAll(".gsel").forEach(b=>{
          b.classList.toggle("selected", b.textContent===d.gesture);
        });
        document.getElementById("status").textContent="Gesture: "+d.gesture;
      });
    }
    function startRecord() {
      fetch("/record/start",{method:"POST"}).then(r=>r.json()).then(d=>{
        document.getElementById("status").textContent=d.message;
        document.getElementById("recBtn").classList.add("active");
      });
    }
    function stopRecord() {
      fetch("/record/stop",{method:"POST"}).then(r=>r.json()).then(d=>{
        document.getElementById("status").textContent=d.message;
        document.getElementById("recBtn").classList.remove("active");
        document.getElementById("saved").textContent=
          "Saved: "+d.saved_count+" templates";
      });
    }
    document.addEventListener("keydown",e=>{
      if(e.key==="r"||e.key==="R") startRecord();
      if(e.key==="s"||e.key==="S") stopRecord();
    });
    setInterval(()=>{
      fetch("/status").then(r=>r.json()).then(d=>{
        document.getElementById("saved").textContent=
          "Saved: "+d.saved_count+" templates for "+d.gesture;
      });
    }, 2000);
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML,
                                  gesture=target_gesture,
                                  gestures=GESTURE_LABELS)


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


@app.route("/set_gesture", methods=["POST"])
def set_gesture():
    global target_gesture
    g = request.get_json().get("gesture", "")
    if g in GESTURE_LABELS:
        target_gesture = g
    return jsonify(gesture=target_gesture)


@app.route("/record/start", methods=["POST"])
def record_start():
    global recording, record_sequence, record_arm_side, last_hand_time
    if not recording:
        recording       = True
        record_sequence = []
        record_arm_side = "unknown"
        last_hand_time  = time.time()
    return jsonify(message=f"Recording {target_gesture}…", recording=True)


@app.route("/record/stop", methods=["POST"])
def record_stop():
    global recording
    if recording:
        finalize_recording()
    return jsonify(message="Saved.", saved_count=saved_count,
                   gesture=target_gesture)


@app.route("/status")
def status():
    return jsonify(gesture=target_gesture, saved_count=saved_count,
                   recording=recording)


def main() -> None:
    global target_gesture
    parser = argparse.ArgumentParser(description="Kinect v2 Gesture Recorder")
    parser.add_argument("--gesture", choices=GESTURE_LABELS,
                        default="SCROLL_UP")
    parser.add_argument("--port", type=int, default=WEB_PORT)
    args = parser.parse_args()
    target_gesture = args.gesture

    log.info(f"Recorder v2 — gesture: {target_gesture}")
    log.info(f"Web UI: http://0.0.0.0:{args.port}")

    threading.Thread(target=capture_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
