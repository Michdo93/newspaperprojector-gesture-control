#!/usr/bin/env python3
"""
Kinect v1 (Xbox 360) Gestensteuerung
Kamera zeigt von oben nach unten
Läuft auf Raspberry Pi 3B+
Sendet Gesten per MQTT an BeagleBone Black
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

# ── Konfiguration ──────────────────────────────────────────────
BROKER  = "192.168.2.XX"   # IP des BBB oder MQTT-Brokers anpassen
PORT    = 1883
TOPIC   = "projektor/geste"

# Mindestbewegung in Pixeln um eine Geste auszulösen
SCHWELLE_WISCHEN   = 40    # px für links/rechts/hoch/runter
SCHWELLE_TIEFE     = 80    # mm Änderung für hoch/runter (Kamera von oben)

# Cooldown zwischen zwei Gesten (Sekunden)
COOLDOWN = 1.2

# Tiefenbereich in dem die Hand erkannt wird (mm)
TIEFE_MIN = 400   # näher als 40cm ignorieren
TIEFE_MAX = 1500  # weiter als 150cm ignorieren
# ──────────────────────────────────────────────────────────────

mqtt_client = mqtt.Client()
mqtt_client.connect(BROKER, PORT, 60)
mqtt_client.loop_start()

letzte_geste_zeit = 0
letzte_hand_x     = None
letzte_hand_y     = None
letzte_hand_tiefe = None  # Durchschnittstiefe der Hand

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
    Kamera zeigt nach unten → Hand die sich hebt = kleinerer Tiefenwert.
    Findet das nächste Objekt im definierten Tiefenbereich.
    Gibt (y, x, mittlere_tiefe) zurück oder None.
    """
    bild = tiefenbild.astype(np.float32)

    # Nur Objekte im sinnvollen Tiefenbereich
    maske = (bild > TIEFE_MIN) & (bild < TIEFE_MAX)
    if not np.any(maske):
        return None

    # Nächsten Punkt finden (kleinster Tiefenwert = nächste Hand)
    bild_gefiltert = np.where(maske, bild, 9999)
    min_pos = np.unravel_index(np.argmin(bild_gefiltert), bild.shape)

    # Region um den nächsten Punkt für stabilere Messung
    y, x = min_pos
    h, w = bild.shape
    y0, y1 = max(0, y-20), min(h, y+20)
    x0, x1 = max(0, x-20), min(w, x+20)
    region = bild[y0:y1, x0:x1]
    region_maske = (region > TIEFE_MIN) & (region < TIEFE_MAX)

    if not np.any(region_maske):
        return None

    mittlere_tiefe = float(np.mean(region[region_maske]))
    return (int(y), int(x), mittlere_tiefe)

def verarbeite_tiefenbild(tiefe, _timestamp):
    global letzte_hand_x, letzte_hand_y, letzte_hand_tiefe

    ergebnis = finde_hand(tiefe)

    if ergebnis is None:
        # Hand nicht sichtbar — Zustand zurücksetzen
        letzte_hand_x     = None
        letzte_hand_y     = None
        letzte_hand_tiefe = None
        return

    hand_y, hand_x, hand_tiefe = ergebnis

    if letzte_hand_x is None:
        # Ersten Frame initialisieren
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

    # Kamera zeigt nach UNTEN:
    # Hand hebt sich → Tiefenwert KLEINER (näher zur Kamera) → HOCH
    # Hand senkt sich → Tiefenwert GRÖSSER (weiter von Kamera) → RUNTER

    if abs_tiefe > SCHWELLE_TIEFE and abs_tiefe > abs_x and abs_tiefe > abs_y:
        # Vertikale Geste (hoch/runter) dominiert
        if delta_tiefe < -SCHWELLE_TIEFE:
            sende_geste("ArrowUp")      # Hand hebt sich
        elif delta_tiefe > SCHWELLE_TIEFE:
            sende_geste("ArrowDown")    # Hand senkt sich

    elif abs_x > SCHWELLE_WISCHEN and abs_x > abs_y:
        # Horizontale Geste (links/rechts) dominiert
        if delta_x > SCHWELLE_WISCHEN:
            sende_geste("ArrowRight")   # Hand nach rechts
        elif delta_x < -SCHWELLE_WISCHEN:
            sende_geste("ArrowLeft")    # Hand nach links

    elif abs_y > SCHWELLE_WISCHEN and abs_y > abs_x:
        # Tiefenrichtung (vorne/hinten aus Kamera-Sicht)
        # Optional: als Scroll verwenden
        pass

    # Aktuelle Position merken
    letzte_hand_x     = hand_x
    letzte_hand_y     = hand_y
    letzte_hand_tiefe = hand_tiefe

log.info(f"Kinect v1 Publisher startet — Broker: {BROKER}:{PORT}")
log.info("Kamera-Ausrichtung: von oben nach unten")
log.info("Gesten: Hand heben=Hoch, Hand senken=Runter, links/rechts=Artikel")
log.info("Strg+C zum Beenden")

try:
    freenect.runloop(depth=verarbeite_tiefenbild)
except KeyboardInterrupt:
    log.info("Beendet.")
finally:
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
