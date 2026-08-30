"""The four target classes NTU cannot teach: sitting, standing, sleeping, attentive.

NTU is built from transitions ("sit down", "stand up") and contains no sustained
states at all, so these cannot be learned from it at any sample count. They are
derived from the skeleton instead.

The derivation deliberately avoids leg geometry. Measured on the reference
auditorium, a hip and knee on the same side are both visible for only 11% of
people — desks hide everything below the waist. Ears are visible for 81-86%, the
nose for 67%. So these rules lean on the head and on how a person changes over
time relative to their own history, which is what survives that occlusion.

Thresholds are starting points calibrated on one scene and one camera angle. They
are the part of the system most in need of checking against real footage.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP, L_KNEE, R_KNEE = 11, 12, 13, 14

STATES_RU = {
    'sitting_attentive': 'сидит, внимателен',
    'sitting': 'сидит',
    'standing': 'стоит',
    'sleeping': 'спит / голова на парте',
    'unknown': 'поза неясна',
}


def head_point(kpts: np.ndarray, scores: np.ndarray, thr: float = 0.3
               ) -> Optional[np.ndarray]:
    """Head position from whichever facial keypoints are visible."""
    idx = [i for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if scores[i] > thr]
    return kpts[idx].mean(axis=0) if idx else None


def head_scale(kpts: np.ndarray, scores: np.ndarray, thr: float = 0.3
               ) -> Optional[float]:
    """A per-person size unit, so thresholds do not depend on distance to camera.

    Ear-to-ear is preferred (ears are the most reliably visible pair); the
    nose-to-shoulder span is the fallback when the head is turned away.
    """
    if scores[L_EAR] > thr and scores[R_EAR] > thr:
        d = np.linalg.norm(kpts[L_EAR] - kpts[R_EAR])
        if d > 1:
            return d * 2.5          # ear span is roughly 0.4 of head height
    shoulders = [i for i in (L_SHOULDER, R_SHOULDER) if scores[i] > thr]
    head = head_point(kpts, scores, thr)
    if shoulders and head is not None:
        d = np.linalg.norm(kpts[shoulders].mean(axis=0) - head)
        if d > 1:
            return d
    return None


def leg_angle(kpts: np.ndarray, scores: np.ndarray, thr: float = 0.3
              ) -> Optional[float]:
    """Hip-knee angle from vertical, when legs happen to be visible.

    Only available for a minority of seated people, but when it is available it
    is the strongest evidence there is, so it takes precedence over the proxies.
    """
    for hip, knee in ((L_HIP, L_KNEE), (R_HIP, R_KNEE)):
        if scores[hip] > thr and scores[knee] > thr:
            v = kpts[knee] - kpts[hip]
            n = np.linalg.norm(v)
            if n > 1:
                # 0 deg = knee directly below hip (standing), 90 = horizontal thigh
                return float(np.degrees(np.arccos(np.clip(v[1] / n, -1, 1))))
    return None


@dataclass
class StateHistory:
    """Per-track memory the rules need — a person judged against their own past."""
    head_y: deque = field(default_factory=lambda: deque(maxlen=90))
    keypoints: deque = field(default_factory=lambda: deque(maxlen=15))
    scale: deque = field(default_factory=lambda: deque(maxlen=90))


class StateClassifier:
    """Turns a tracked person into one of the sustained-posture classes."""

    def __init__(self,
                 stand_ratio: float = 4.5,
                 head_drop: float = 1.2,
                 still_motion: float = 0.06,
                 sleep_frames: int = 20,
                 leg_stand_deg: float = 35.0,
                 min_history: int = 10):
        self.stand_ratio = stand_ratio
        self.head_drop = head_drop
        self.still_motion = still_motion
        self.sleep_frames = sleep_frames
        self.leg_stand_deg = leg_stand_deg
        self.min_history = min_history
        self.history: Dict[int, StateHistory] = {}

    def drop(self, track_id: int):
        self.history.pop(track_id, None)

    def _motion(self, hist: StateHistory, scale: float) -> Optional[float]:
        """Median joint movement between recent frames, in head-sizes."""
        if len(hist.keypoints) < 3 or not scale:
            return None
        arr = np.stack(hist.keypoints)
        steps = np.linalg.norm(np.diff(arr, axis=0), axis=-1)
        return float(np.median(steps) / scale)

    def __call__(self, track_id: int, box: np.ndarray, kpts: np.ndarray,
                 scores: np.ndarray) -> Tuple[str, float]:
        hist = self.history.setdefault(track_id, StateHistory())

        scale = head_scale(kpts, scores)
        head = head_point(kpts, scores)
        if scale:
            hist.scale.append(scale)
        if head is not None:
            hist.head_y.append(float(head[1]))
        hist.keypoints.append(kpts.copy())

        if scale is None or head is None:
            return 'unknown', 0.0

        # --- sleeping: head parked below its own usual level, and still ---
        # Writing also lowers the head, so stillness is what separates them: a
        # writing hand keeps moving, a sleeping one does not.
        if len(hist.head_y) >= self.min_history:
            baseline = float(np.percentile(hist.head_y, 25))
            drop = (head[1] - baseline) / scale
            motion = self._motion(hist, scale)
            recent_low = sum(1 for y in list(hist.head_y)[-self.sleep_frames:]
                             if (y - baseline) / scale > self.head_drop * 0.6)
            if (drop > self.head_drop and motion is not None
                    and motion < self.still_motion
                    and recent_low >= self.sleep_frames * 0.6):
                return 'sleeping', min(drop / self.head_drop, 2.0) / 2.0

        # --- standing vs sitting ---
        angle = leg_angle(kpts, scores)
        if angle is not None:
            standing = angle < self.leg_stand_deg
            return ('standing' if standing else 'sitting'), 0.9

        # Legs hidden: how much body is visible above the desk, in head-sizes.
        # A standing person shows most of their height; a seated one is cut off.
        ratio = (box[3] - box[1]) / scale
        if ratio > self.stand_ratio:
            return 'standing', min((ratio - self.stand_ratio) / 2 + 0.5, 0.9)
        return 'sitting', min((self.stand_ratio - ratio) / 2 + 0.5, 0.9)

    def refine_attentive(self, state: str, objects, head_down: bool) -> str:
        """"Attentive" is sitting with nothing else going on.

        It is a residual class rather than a physical property, so it is decided
        last, from the absence of other evidence.
        """
        if state != 'sitting':
            return state
        if objects or head_down:
            return 'sitting'
        return 'sitting_attentive'
