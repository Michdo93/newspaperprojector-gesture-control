"""
DTW (Dynamic Time Warping) gesture matcher.

Feature vector design:
  Each frame stores the CUMULATIVE DISPLACEMENT from the gesture start:
    [delta_cx, delta_cy, delta_depth]
  All three are differences from the reference position (first frame).
  This makes the representation:
    - Translation-invariant: absolute position does not matter
    - Direction-aware: SCROLL_UP has negative delta_depth, SCROLL_DOWN positive
    - Robust to Hin+Rückbewegung if only the peak (Hinbewegung) is recorded

Template format: NumPy .npy files, shape (N, 3), dtype float32.
"""

import os
import glob
import numpy as np
import logging
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import TEMPLATE_DIR, DTW_MAX_DISTANCE

log = logging.getLogger("dtw_matcher")

GESTURE_LABELS = ["SCROLL_UP", "SCROLL_DOWN", "PAGE_NEXT", "PAGE_PREV"]


def _dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Normalised DTW distance between two sequences of shape (N,D) and (M,D)."""
    n, m = len(seq_a), len(seq_b)
    dtw  = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost      = float(np.linalg.norm(seq_a[i - 1] - seq_b[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j],
                                   dtw[i, j - 1],
                                   dtw[i - 1, j - 1])
    return dtw[n, m] / (n + m)


def _to_cumulative(seq: np.ndarray) -> np.ndarray:
    """
    Convert an absolute position sequence to a cumulative displacement sequence.
    Input shape: (N, 3) — [norm_cx, norm_cy, norm_depth] per frame.
    Output shape: (N, 3) — displacement from the first frame.
    Frame 0 is always [0, 0, 0].
    """
    return seq - seq[0]


class GestureMatcher:
    """
    Loads all .npy templates and matches incoming sequences via DTW.
    Multiple templates per gesture are supported.
    """

    def __init__(self, template_dir: str | None = None):
        self.template_dir = template_dir or TEMPLATE_DIR
        self.templates: dict[str, list[np.ndarray]] = {g: [] for g in GESTURE_LABELS}
        self._load_templates()

    def _load_templates(self) -> None:
        total = 0
        for label in GESTURE_LABELS:
            folder = os.path.join(self.template_dir, label)
            if not os.path.isdir(folder):
                continue
            for path in glob.glob(os.path.join(folder, "*.npy")):
                seq = np.load(path)
                if seq.ndim == 2 and seq.shape[1] == 3 and len(seq) >= 2:
                    self.templates[label].append(_to_cumulative(seq))
                    total += 1
        log.info(f"Loaded {total} gesture templates from '{self.template_dir}'")
        for label, seqs in self.templates.items():
            log.info(f"  {label}: {len(seqs)} template(s)")

    def reload(self) -> None:
        self.templates = {g: [] for g in GESTURE_LABELS}
        self._load_templates()

    def match(self, sequence: list[np.ndarray],
              arm_side: str = "unknown") -> "tuple[str | None, float]":
        """
        Match a list of position vectors against all templates.

        Parameters
        ----------
        sequence : list of np.ndarray, each shape (3,)
            Raw [norm_cx, norm_cy, norm_depth] per frame since gesture start.
        arm_side : str — logged but does not filter gesture labels.

        Returns
        -------
        (gesture_label, distance) or (None, inf) if no match.
        """
        if len(sequence) < 2:
            return None, float("inf")

        obs = _to_cumulative(np.stack(sequence))
        best_label    = None
        best_distance = float("inf")

        for label, tmpl_list in self.templates.items():
            for tmpl in tmpl_list:
                dist = _dtw_distance(obs, tmpl)
                if dist < best_distance:
                    best_distance = dist
                    best_label    = label

        if best_distance <= DTW_MAX_DISTANCE:
            log.debug(f"Matched '{best_label}' dist={best_distance:.4f} arm={arm_side}")
            return best_label, best_distance

        log.debug(f"No match — best='{best_label}' dist={best_distance:.4f}")
        return None, best_distance
