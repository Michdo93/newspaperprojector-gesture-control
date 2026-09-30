"""
Shared Kinect v1 depth-frame processing utilities.
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
                 centroid_x: int, centroid_y: int, pixel_count: int):
        self.y           = y             # closest-point row
        self.x           = x             # closest-point column
        self.mean_depth  = mean_depth    # mean depth of the region (mm)
        self.centroid_x  = centroid_x    # x centroid of the full hand blob
        self.centroid_y  = centroid_y    # y centroid of the full hand blob
        self.pixel_count = pixel_count   # number of valid pixels in the blob
        self.arm_side    = self._arm_side()

    def _arm_side(self) -> str:
        """
        Determine arm side from centroid x position.
        In CEILING mode the image may be mirrored, so the split is
        relative to the (possibly mirrored) frame width.
        Returns "right" or "left" (from the user's perspective).
        """
        # We do not have the frame width here, so arm_side is finalised
        # by detect_hand() once the width is known.
        return "unknown"

    def as_feature_vector(self) -> np.ndarray:
        """
        Return a compact feature vector for DTW matching.
        Uses relative (normalised) coordinates so translation does not matter.
        Format: [norm_centroid_x, norm_centroid_y, norm_depth]
        All values in [0, 1].
        """
        return np.array([
            self.centroid_x / 640.0,
            self.centroid_y / 480.0,
            (self.mean_depth - DEPTH_MIN) / max(1, DEPTH_MAX_HAND - DEPTH_MIN)
        ], dtype=np.float32)


def detect_hand(depth_frame: np.ndarray) -> HandDetection | None:
    """
    Locate the closest hand-sized object in the depth frame.
    Returns a HandDetection or None if no valid hand is found.
    """
    img = depth_frame.astype(np.float32)

    valid_mask = (img > DEPTH_MIN) & (img < DEPTH_MAX_HAND)
    if not np.any(valid_mask):
        return None
    if int(np.sum(valid_mask)) < MIN_HAND_PIXELS:
        return None

    # Closest point
    search_img = np.where(valid_mask, img, 9999.0)
    min_pos    = np.unravel_index(np.argmin(search_img), img.shape)
    py, px     = int(min_pos[0]), int(min_pos[1])

    # Region around the closest point
    h, w   = img.shape
    y0, y1 = max(0, py - 25), min(h, py + 25)
    x0, x1 = max(0, px - 25), min(w, px + 25)
    region      = img[y0:y1, x0:x1]
    region_mask = (region > DEPTH_MIN) & (region < DEPTH_MAX_HAND)

    if not np.any(region_mask):
        return None
    if int(np.sum(region_mask)) < MIN_HAND_PIXELS:
        return None

    mean_depth = float(np.mean(region[region_mask]))

    # Centroid of the full valid blob (for arm-side detection)
    ys, xs     = np.where(valid_mask)
    centroid_x = int(np.mean(xs))
    centroid_y = int(np.mean(ys))
    pixel_count = int(np.sum(valid_mask))

    det = HandDetection(py, px, mean_depth, centroid_x, centroid_y, pixel_count)

    # Finalise arm side (right = user's right, accounting for mirror)
    if MIRROR_X:
        det.arm_side = "right" if centroid_x > w * ARM_SPLIT_RATIO else "left"
    else:
        det.arm_side = "left" if centroid_x > w * ARM_SPLIT_RATIO else "right"

    return det


def build_heatmap(depth_frame: np.ndarray) -> np.ndarray:
    """
    Convert a depth frame to a BGR colour heatmap for display.
    - Coloured (JET, blue=far red=close): within hand detection zone
    - Dark grey: background (in DEPTH_MAX range but beyond DEPTH_MAX_HAND)
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
