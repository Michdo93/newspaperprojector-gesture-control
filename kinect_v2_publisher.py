#!/usr/bin/env python3
"""
Kinect v2 (Xbox One) Gestensteuerung
Kamera zeigt von oben nach unten
Läuft auf Raspberry Pi 3B+ (USB 3.0 nötig — Pi 3B+ hat nur USB 2.0!)
WARNUNG: Kinect v2 braucht USB 3.0 — Pi 3B+ hat nur USB 2.0.
         Empfehlung: Kinect v1 für Pi 3B+ verwenden!
         Dieses Script nur für Pi 4 oder Pi 5 mit USB 3.0.
Sendet Gesten per MQTT an BeagleBone Black
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

# ── Konfiguration ──────────────────────────────────────────────
BROKER  = "192.168.2.XX"   # IP des BBB oder MQTT-Brokers anpassen
PORT    = 1883
TOPIC   = "projektor/geste"

SCHWELLE_WISCHEN   = 50    # px für links/rechts
SCHWELLE_TIEFE     = 100   # mm Änderung für hoch/runter
COOLDOWN           = 1.2   # Sekunden zwischen Gesten

TIEFE_MIN = 500            # mm
TIEFE_MAX = 2000           # mm
# ──────────────────────────────────────────────────────────────

try:
    from pylibfreenect2 import Freenect2, SyncMultiFrameListener
    from pylibfreenect2 import FrameType, Registration, Frame
    import pylibfreenect2 as fn2
except ImportError:
    log.error("pylibfreenect2 nicht installiert!")
    log.error("sudo pip3 install pylibfreenect2 --break-system-packages")
    exit(1)

mqtt_client = mqtt.Client()
mqtt_client.connect(BROKER, PORT, 60)
mqtt_client.loop_start()

letzte_geste_zeit = 0
letzte_hand_x     = None
letzte_hand_y     = None
letzte_hand_tiefe = None

def sende_geste(geste: str):
    global letzte_geste_zeit
    jetzt = time.time()
    if jetzt - letzte_geste_zeit < COOLDOWN:
        return
    letzte_geste_zeit = jetzt
    mqtt_client.publish(TOPIC, geste)
    log.info(f"Geste gesendet: {geste}")

def finde_hand(tiefenbild: np.ndarray):
    """
    Findet die nächste Hand im Tiefenbild.
    Gibt (y, x, mittlere_tiefe) zurück oder None.
    """
    bild = tiefenbild.astype(np.float32)
    maske = (bild > TIEFE_MIN) & (bild < TIEFE_MAX)
    if not np.any(maske):
        return None

    bild_gefiltert = np.where(maske, bild, 99999)
    min_pos = np.unravel_index(np.argmin(bild_gefiltert), bild.shape)

    y, x = min_pos
    h, w = bild.shape
    y0, y1 = max(0, y-25), min(h, y+25)
    x0, x1 = max(0, x-25), min(w, x+25)
    region = bild[y0:y1, x0:x1]
    region_maske = (region > TIEFE_MIN) & (region < TIEFE_MAX)

    if not np.any(region_maske):
        return None

    mittlere_tiefe = float(np.mean(region[region_maske]))
    return (int(y), int(x), mittlere_tiefe)

def verarbeite_frame(tiefenbild: np.ndarray):
    global letzte_hand_x, letzte_hand_y, letzte_hand_tiefe

    ergebnis = finde_hand(tiefenbild)

    if ergebnis is None:
        letzte_hand_x     = None
        letzte_hand_y     = None
        letzte_hand_tiefe = None
        return

    hand_y, hand_x, hand_tiefe = ergebnis

    if letzte_hand_x is None:
        letzte_hand_x     = hand_x
        letzte_hand_y     = hand_y
        letzte_hand_tiefe = hand_tiefe
        return

    delta_x     = hand_x     - letzte_hand_x
    delta_y     = hand_y     - letzte_hand_y
    delta_tiefe = hand_tiefe - letzte_hand_tiefe

    abs_x     = abs(delta_x)
    abs_y     = abs(delta_y)
    abs_tiefe = abs(delta_tiefe)

    if abs_tiefe > SCHWELLE_TIEFE and abs_tiefe > abs_x and abs_tiefe > abs_y:
        if delta_tiefe < -SCHWELLE_TIEFE:
            sende_geste("ArrowUp")
        elif delta_tiefe > SCHWELLE_TIEFE:
            sende_geste("ArrowDown")
    elif abs_x > SCHWELLE_WISCHEN and abs_x > abs_y:
        if delta_x > SCHWELLE_WISCHEN:
            sende_geste("ArrowRight")
        elif delta_x < -SCHWELLE_WISCHEN:
            sende_geste("ArrowLeft")

    letzte_hand_x     = hand_x
    letzte_hand_y     = hand_y
    letzte_hand_tiefe = hand_tiefe

# ── Kinect v2 initialisieren ───────────────────────────────────
log.info("Kinect v2 Publisher startet")
log.warning("WARNUNG: Pi 3B+ hat nur USB 2.0 — Kinect v2 braucht USB 3.0!")
log.warning("Empfehlung: Kinect v1 für Pi 3B+ verwenden")

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    log.error("Keine Kinect v2 gefunden!")
    exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial)

listener = SyncMultiFrameListener(FrameType.Depth)
device.setIrAndDepthFrameListener(listener)
device.start()

log.info(f"Kinect v2 verbunden — Broker: {BROKER}:{PORT}")
log.info("Strg+C zum Beenden")

try:
    while True:
        frames = listener.waitForNewFrame()
        depth_frame = frames["depth"]
        tiefenbild = depth_frame.asarray()
        verarbeite_frame(tiefenbild)
        listener.release(frames)

except KeyboardInterrupt:
    log.info("Beendet.")
finally:
    device.stop()
    device.close()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
