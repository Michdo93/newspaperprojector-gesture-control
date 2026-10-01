"""
MQTT connection helper — shared by all scripts.
Reads credentials and TLS settings from config/settings.py.
"""

import ssl
import paho.mqtt.client as mqtt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
    MQTT_CA_CERT, MQTT_TOPIC
)


def build_mqtt_client(client_id: str = "gesture_control") -> mqtt.Client:
    """
    Create and connect a Paho MQTT client.
    Uses TLS if MQTT_CA_CERT is set, plain TCP otherwise.
    Returns a connected client with loop_start() already called.
    """
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)

    if MQTT_USERNAME:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD or None)

    if MQTT_CA_CERT:
        client.tls_set(
            ca_certs=MQTT_CA_CERT,
            tls_version=ssl.PROTOCOL_TLS_CLIENT
        )
        port = 8883
    else:
        port = MQTT_PORT

    client.connect(MQTT_BROKER, port, keepalive=60)
    client.loop_start()
    return client
