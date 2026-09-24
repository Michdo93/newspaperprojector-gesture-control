# newspaperprojector-gesture-control
Gesture control for the [newspaperprojector](https://github.com/Michdo93/newspaperprojector) using an Xbox Kinect camera and a Raspberry Pi.

## Installation on the Raspberry Pi 3B+

### For Kinect v1 (recommended for Pi 3B+)

```
# Updating the System
sudo apt-get update && sudo apt-get upgrade -y

# Dependencies
sudo apt-get install -y \
  libfreenect-dev \
  libfreenect0.5 \
  freenect \
  python3-pip \
  python3-numpy

# Python Libraries
sudo pip3 install freenect paho-mqtt --break-system-packages

# Make Kinect usable without root
sudo adduser pi plugdev
sudo adduser pi video

# Creating udev Rules for Kinect
sudo nano /etc/udev/rules.d/51-kinect.rules
```

Content of the udev rule:

```
# Kinect v1 - NUI Motor
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02c2", MODE="0666"

# Kinect v1 - NUI Audio
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02ad", MODE="0666"

# Kinect v1 - NUI Camera (Tiefensensor)
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02ae", MODE="0666"
```

```
sudo udevadm control --reload-rules

# Testen ob Kinect erkannt wird
lsusb | grep Microsoft
freenect-glview  # Sichttest — zeigt Tiefenbild
```

### For Kinect v2 (Pi 4 or Pi 5 only!)

```
# Updating the System
sudo apt-get update && sudo apt-get upgrade -y

# Dependencies
sudo apt-get install -y \
  libfreenect2-dev \
  python3-pip \
  python3-numpy

# Python Libraries
sudo pip3 install pylibfreenect2 paho-mqtt --break-system-packages

# udev Rules
sudo apt-get install -y libusb-1.0-0-dev
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02d8", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/51-kinect2.rules
sudo udevadm control --reload-rules
```

### Autostart on the Pi

```
# Copy the Script
mkdir -p /home/pi/scripts
# On Raspberry Pi 3b+
cp kinect_v1_publisher.py /home/pi/scripts/kinect_publisher.py
# On Raspberry Pi 4 or 5
cp kinect_v2_publisher.py /home/pi/scripts/kinect_publisher.py

# Creating Systemd Service
sudo nano /etc/systemd/system/kinect-publisher.service
```

```
[Unit]
Description=Kinect Gesten Publisher
After=network.target

[Service]
User=pi
ExecStart=/usr/bin/python3 /home/pi/scripts/kinect_publisher.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```
sudo systemctl enable kinect-publisher
sudo systemctl start kinect-publisher
sudo systemctl status kinect-publisher
```

> [!CAUTION]
> **Important Note: Pi 3B+ and Kinect v2**
> The **Pi 3B+ only has USB 2.0** — the Kinect v2 absolutely requires USB 3.0 for the depth image bandwidth. With USB 2.0, it either doesn't run stably or doesn't work at all. **The Kinect v1 is the right choice for the Pi 3B+.**

## Usage

| Swipe Direction | MQTT Topic | Message Payload | Gesture Description | Control Description |
| --- | --- | --- | --- | --- |
| **Swipe Right** | `projector/command/gesture` | `PAGE_NEXT` | Triggered when the hand moves horizontally to the right across the camera frame. | Go to the next article in the daily newspaper. |
| **Swipe Left** | `projector/command/gesture` | `PAGE_PREV` | Triggered when the hand moves horizontally to the left across the camera frame. | Go to the previous article in the daily newspaper. |
| **Swipe Up** | `projector/command/gesture` | `SCROLL_UP` | Triggered when raising the hand towards the top-down camera (depth distance decreases). | Scroll up on the daily newspaper's website. |
| **Swipe Down** | `projector/command/gesture` | `SCROLL_DOWN` | Triggered when lowering the hand away from the top-down camera (depth distance increases). | Scroll down on the daily newspaper's website. |













