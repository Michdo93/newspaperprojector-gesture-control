"""
Shared Kinect v1/v2 depth-frame processing utilities.
Used by the recorder, web viewer and headless publisher.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    DEPTH_MIN, DEPTH_MAX, DEPTH_MAX_HAND, MIN_HAND_PIXELS,
    CAMERA_MODE, MIRROR_X, ARM_SPLIT_RATIO
)


class HandDetection:
    """Result of a single-frame hand detection."""

    def __init__(self, y: int, x: int, mean_depth: float,
                 centroid_x: int, centroid_y: int,
                 pixel_count: int, frame_w: int, frame_h: int):
        self.y           = y
        self.x           = x
        self.mean_depth  = mean_depth
        self.centroid_x  = centroid_x
        self.centroid_y  = centroid_y
        self.pixel_count = pixel_count
        self.norm_cx     = centroid_x / max(1, frame_w)
        self.norm_cy     = centroid_y / max(1, frame_h)
        self.norm_depth  = (mean_depth - DEPTH_MIN) / max(1, DEPTH_MAX_HAND - DEPTH_MIN)
        self.arm_side    = self._arm_side(frame_w)

    def _arm_side(self, frame_w: int) -> str:
        if MIRROR_X:
            return "right" if self.centroid_x > frame_w * ARM_SPLIT_RATIO else "left"
        else:
            return "left" if self.centroid_x > frame_w * ARM_SPLIT_RATIO else "right"

    def position(self) -> np.ndarray:
        """
        Return the current position as a 3-element vector:
          [norm_cx, norm_cy, norm_depth]
        All values in [0, 1].
        Used to compute cumulative displacement from the reference position.
        """
        return np.array([self.norm_cx, self.norm_cy, self.norm_depth],
                        dtype=np.float32)


def detect_hand(depth_frame: np.ndarray) -> "HandDetection | None":
    """
    Locate the closest hand-sized object in the depth frame.
    Returns a HandDetection or None if no valid hand is found.
    """
    img = depth_frame.astype(np.float32)
    h, w = img.shape

    valid_mask = (img > DEPTH_MIN) & (img < DEPTH_MAX_HAND)
    if not np.any(valid_mask):
        return None
    if int(np.sum(valid_mask)) < MIN_HAND_PIXELS:
        return None

    search_img = np.where(valid_mask, img, 9999.0)
    min_pos    = np.unravel_index(np.argmin(search_img), img.shape)
    py, px     = int(min_pos[0]), int(min_pos[1])

    y0, y1 = max(0, py - 25), min(h, py + 25)
    x0, x1 = max(0, px - 25), min(w, px + 25)
    region      = img[y0:y1, x0:x1]
    region_mask = (region > DEPTH_MIN) & (region < DEPTH_MAX_HAND)

    if not np.any(region_mask):
        return None
    if int(np.sum(region_mask)) < MIN_HAND_PIXELS:
        return None

    mean_depth = float(np.mean(region[region_mask]))

    ys, xs      = np.where(valid_mask)
    centroid_x  = int(np.mean(xs))
    centroid_y  = int(np.mean(ys))
    pixel_count = int(np.sum(valid_mask))

    return HandDetection(py, px, mean_depth,
                         centroid_x, centroid_y,
                         pixel_count, w, h)


def build_heatmap(depth_frame: np.ndarray) -> np.ndarray:
    """
    Convert a depth frame to a BGR colour heatmap for display.
    - Coloured (JET): within hand detection zone
    - Dark grey: background (beyond DEPTH_MAX_HAND but within DEPTH_MAX)
    - Near-black: out of range entirely
    """
    import cv2
    img = depth_frame.astype(np.float32)

    in_hand_zone  = (img > DEPTH_MIN) & (img < DEPTH_MAX_HAND)
    in_full_range = (img > DEPTH_MIN) & (img < DEPTH_MAX)

    display = np.zeros_like(img)
    display[in_hand_zone] = (
        (img[in_hand_zone] - DEPTH_MIN) /
        max(1, DEPTH_MAX_HAND - DEPTH_MIN) * 255.0
    )
    display = display.astype(np.uint8)

    colour = cv2.applyColorMap(255 - display, cv2.COLORMAP_JET)
    colour[in_full_range & ~in_hand_zone] = [55, 55, 55]
    colour[~in_full_range]               = [18, 18, 18]
    return colour
