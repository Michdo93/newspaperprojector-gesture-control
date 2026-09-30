"""
DTW (Dynamic Time Warping) gesture matcher.
Loads template sequences from the templates/ directory and matches
incoming gesture sequences against them.

Template format: NumPy .npy files, shape (N, 3), dtype float32.
Each row is [norm_centroid_x, norm_centroid_y, norm_depth].
Translation invariance: features are relative (normalised coordinates +
depth delta), so the user can stand anywhere in the frame.
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
    """
    Compute the normalised DTW distance between two feature sequences.
    Both sequences have shape (N, D) and (M, D).
    Returns a scalar in [0, inf); lower = more similar.
    """
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


def _to_delta_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Convert an absolute feature sequence to a delta (difference) sequence.
    This makes matching translation-invariant in the feature space.
    Shape: (N-1, D) from input (N, D).
    """
    if len(seq) < 2:
        return seq
    return np.diff(seq, axis=0)


class GestureMatcher:
    """
    Loads all .npy templates and matches incoming sequences via DTW.
    Multiple templates per gesture are supported — the minimum distance
    across all templates for a label is used.
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
                    self.templates[label].append(_to_delta_sequence(seq))
                    total += 1
        log.info(f"Loaded {total} gesture templates from '{self.template_dir}'")
        for label, seqs in self.templates.items():
            log.info(f"  {label}: {len(seqs)} template(s)")

    def reload(self) -> None:
        """Reload templates from disk (call after recording new ones)."""
        self.templates = {g: [] for g in GESTURE_LABELS}
        self._load_templates()

    def match(self, sequence: list[np.ndarray],
              arm_side: str = "unknown") -> tuple[str | None, float]:
        """
        Match a list of feature vectors against all templates.

        Parameters
        ----------
        sequence : list of np.ndarray, each shape (3,)
            The observed gesture sequence (raw feature vectors).
        arm_side : str
            "left", "right", or "unknown" — logged but not used for filtering
            (same gestures are valid for both arms).

        Returns
        -------
        (gesture_label, distance) or (None, inf) if no match found.
        """
        if len(sequence) < 2:
            return None, float("inf")

        obs_delta = _to_delta_sequence(np.stack(sequence))
        best_label    = None
        best_distance = float("inf")

        for label, tmpl_list in self.templates.items():
            for tmpl in tmpl_list:
                dist = _dtw_distance(obs_delta, tmpl)
                if dist < best_distance:
                    best_distance = dist
                    best_label    = label

        if best_distance <= DTW_MAX_DISTANCE:
            log.debug(f"Matched '{best_label}' (dist={best_distance:.3f}, arm={arm_side})")
            return best_label, best_distance

        log.debug(f"No match (best={best_label}, dist={best_distance:.3f})")
        return None, best_distance
