"""
Central configuration for the Newspaper Projector Gesture Control system.
Edit this file to match your hardware setup and MQTT broker.
"""

# ── MQTT ──────────────────────────────────────────────────────────────────────
MQTT_BROKER   = "127.0.0.1"        # Change to BBB IP for production
MQTT_PORT     = 1883               # 1883 = local unencrypted, 8883 = TLS
MQTT_USERNAME = "projector"
MQTT_PASSWORD = ""                 # Set to your Mosquitto password

# TLS — leave empty string to disable TLS (local/testing broker)
# For production: copy ca.crt from BBB:
#   scp debian@newspaperprojector.local:/etc/mosquitto/certs/ca.crt ~/
MQTT_CA_CERT  = ""                 # e.g. "/home/pi/ca.crt"

MQTT_TOPIC    = "projector/command/gesture"

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_MODE = "CEILING"            # "CEILING" or "DESK"
MIRROR_X    = True                 # flip horizontal if gestures appear reversed

# ── Depth detection zone ──────────────────────────────────────────────────────
# Objects within DEPTH_MIN..DEPTH_MAX_HAND are candidates for hand detection.
# Objects beyond DEPTH_MAX_HAND are treated as background.
# Calibrate using the web viewer — check the HUD depth statistics.
DEPTH_MIN      = 600               # mm
DEPTH_MAX      = 2000              # mm — absolute cutoff
DEPTH_MAX_HAND = 870               # mm — background threshold

# ── Hand detection ────────────────────────────────────────────────────────────
MIN_HAND_PIXELS = 150
STABLE_FRAMES   = 8               # frames hand must be continuously visible

# ── Template matching (DTW) ───────────────────────────────────────────────────
TEMPLATE_DIR          = "templates"
DTW_MAX_DISTANCE      = 0.35      # normalised DTW distance threshold
MIN_GESTURE_FRAMES    = 8         # minimum frames to form a valid gesture
MAX_GESTURE_FRAMES    = 60        # maximum frames captured per gesture
GESTURE_END_TIMEOUT   = 0.8       # seconds without hand → gesture sequence ends

# Arm side: centroid x < ARM_SPLIT_RATIO * width  →  right arm (in CEILING mode)
ARM_SPLIT_RATIO = 0.5

# ── Cooldown ──────────────────────────────────────────────────────────────────
COOLDOWN = 1.0                     # seconds between published gestures

# ── Web viewer ────────────────────────────────────────────────────────────────
WEB_PORT     = 8080
JPEG_QUALITY = 70
STREAM_FPS   = 25
