"""Attaches identities to tracked people, whatever identity system is plugged in.

Two facts shape this design.

The first is that the tracker already solves the hard part. A person keeps a
stable id across frames, so identity has to be resolved *once per track*, not
once per frame — the recognition model runs a few times per person per session
instead of thirty times a second.

The second is measured, and less comfortable: on one wide shot of a lecture hall
the median distance between a person's eyes is about 10 px, where face
recognition generally wants 30+. Only a few percent of a full hall can be
identified at all from such a frame. So an unidentified person is the normal
case here, not an error — nothing downstream may assume a name exists.
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

L_EYE, R_EYE, L_EAR, R_EAR, NOSE = 1, 2, 3, 4, 0


class IdentityProvider(Protocol):
    """What an identity system has to expose to be usable here.

    Return (identity, confidence) or None when the person cannot be recognised.
    Returning None is expected and must not raise.
    """

    def identify(self, frame: np.ndarray, box: np.ndarray,
                 keypoints: np.ndarray, kpt_scores: np.ndarray
                 ) -> Optional[Tuple[str, float]]:
        ...


def eye_distance(keypoints: np.ndarray, kpt_scores: np.ndarray,
                 thr: float = 0.3) -> Optional[float]:
    """Inter-ocular distance in pixels — the usual proxy for "is the face big
    enough to recognise".

    Deliberately requires both eyes rather than falling back to the ear span. In
    a hall shot from behind the audience, both ears are visible on the back of a
    head while no face is: measured on the reference scene, an ear-based estimate
    passed 63% of people as recognisable where the eyes said 4%. Ear span is a
    head-width measurement, not a face-visibility one, and using it here would
    send the recognition model a stream of backs of heads.

    Returning None means "cannot tell", which the caller treats as not eligible.
    """
    if kpt_scores[L_EYE] > thr and kpt_scores[R_EYE] > thr:
        return float(np.linalg.norm(keypoints[L_EYE] - keypoints[R_EYE]))
    return None


def face_crop(frame: np.ndarray, keypoints: np.ndarray, kpt_scores: np.ndarray,
              margin: float = 1.8, thr: float = 0.3) -> Optional[np.ndarray]:
    """Cut the face out using the head keypoints the pose model already gives.

    Saves running a separate face detector: the head points are computed anyway,
    and the crop they define is what most recognition models want as input.
    """
    idx = [i for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if kpt_scores[i] > thr]
    if len(idx) < 2:
        return None
    pts = keypoints[idx]
    centre = pts.mean(axis=0)
    span = max(float(np.ptp(pts, axis=0).max()), 1.0) * margin
    h, w = frame.shape[:2]
    x1, y1 = int(max(centre[0] - span, 0)), int(max(centre[1] - span, 0))
    x2, y2 = int(min(centre[0] + span, w)), int(min(centre[1] + span, h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]


class FaceCropAdapter:
    """Wraps a model that takes a face image: `fn(face_bgr) -> (id, conf)|None`."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def identify(self, frame, box, keypoints, kpt_scores):
        face = face_crop(frame, keypoints, kpt_scores)
        return self.fn(face) if face is not None else None


class PersonCropAdapter:
    """Wraps a model that takes the whole person: `fn(person_bgr) -> (id, conf)|None`.

    This is the shape appearance-based re-identification takes — it does not need
    a visible face, which matters when most faces in the hall are a few pixels
    wide. It answers "the same person as before", not "who".
    """

    def __init__(self, fn: Callable):
        self.fn = fn

    def identify(self, frame, box, keypoints, kpt_scores):
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        h, w = frame.shape[:2]
        crop = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
        return self.fn(crop) if crop.size else None


class FrameAdapter:
    """Wraps a system that scans a whole frame and returns its own boxes.

    `fn(frame) -> [(box_xyxy, identity, confidence), ...]`. Results are matched to
    tracked people by box overlap, so the two systems stay independent.
    """

    def __init__(self, fn: Callable, iou_thr: float = 0.3):
        self.fn = fn
        self.iou_thr = iou_thr
        self._cache_frame_id = None
        self._cache = []

    def _detections(self, frame):
        key = id(frame)
        if key != self._cache_frame_id:
            self._cache_frame_id = key
            self._cache = self.fn(frame) or []
        return self._cache

    def identify(self, frame, box, keypoints, kpt_scores):
        best, best_iou = None, self.iou_thr
        for other, name, conf in self._detections(frame):
            iou = _iou(box[:4], np.asarray(other, dtype=float))
            if iou > best_iou:
                best, best_iou = (name, conf), iou
        return best


def _iou(a, b) -> float:
    lt = np.maximum(a[:2], b[:2])
    rb = np.minimum(a[2:4], b[2:4])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[0] * wh[1]
    area = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / area) if area > 0 else 0.0


@dataclass
class TrackIdentity:
    name: Optional[str] = None
    confidence: float = 0.0
    attempts: int = 0
    last_attempt: float = 0.0
    resolved_at: float = 0.0


class IdentityResolver:
    """Resolves identity per track, sparingly, and remembers the answer.

    Retries are deliberately limited. A person seated with their head down will
    never be recognised no matter how many times the model is asked, and asking
    on every frame would cost more than the rest of the pipeline combined.
    """

    def __init__(self, provider: IdentityProvider,
                 min_eye_px: float = 25.0,
                 min_confidence: float = 0.6,
                 max_attempts: int = 8,
                 retry_seconds: float = 3.0,
                 refresh_seconds: Optional[float] = None):
        self.provider = provider
        # Below this the face carries too few pixels to be worth a forward pass.
        self.min_eye_px = min_eye_px
        self.min_confidence = min_confidence
        self.max_attempts = max_attempts
        self.retry_seconds = retry_seconds
        # Set to re-check a name occasionally, in case a track drifted onto the
        # wrong person after a long occlusion.
        self.refresh_seconds = refresh_seconds
        self.identities: Dict[int, TrackIdentity] = {}
        self.stats = dict(attempted=0, resolved=0, too_small=0)

    def drop(self, track_id: int):
        self.identities.pop(track_id, None)

    def _should_try(self, rec: TrackIdentity, now: float) -> bool:
        if rec.name is not None:
            if self.refresh_seconds is None:
                return False
            return now - rec.resolved_at > self.refresh_seconds
        if rec.attempts >= self.max_attempts:
            return False
        return now - rec.last_attempt > self.retry_seconds

    def resolve(self, frame: np.ndarray, people) -> Dict[int, TrackIdentity]:
        """Fill in identities for the given people; returns the current table."""
        now = time.time()
        for p in people:
            rec = self.identities.setdefault(p.track_id, TrackIdentity())
            if not self._should_try(rec, now):
                continue

            # No measurable eye distance means the face is turned away or too
            # degraded — not a candidate. Treating "unknown" as eligible would
            # spend the whole budget on the backs of heads.
            eye = eye_distance(p.keypoints, p.kpt_scores)
            if eye is None or eye < self.min_eye_px:
                self.stats['too_small'] += 1
                rec.last_attempt = now
                continue

            rec.attempts += 1
            rec.last_attempt = now
            self.stats['attempted'] += 1
            try:
                result = self.provider.identify(frame, p.box, p.keypoints,
                                                p.kpt_scores)
            except Exception:
                # A failing identity model must not take the pipeline down with
                # it; the person simply stays anonymous.
                result = None
            if result:
                name, conf = result
                if conf >= self.min_confidence:
                    rec.name, rec.confidence, rec.resolved_at = name, conf, now
                    self.stats['resolved'] += 1
        return self.identities

    def get(self, track_id: int) -> TrackIdentity:
        return self.identities.get(track_id, TrackIdentity())

    def report(self) -> str:
        known = sum(1 for r in self.identities.values() if r.name)
        total = len(self.identities)
        s = self.stats
        return (f"опознано {known}/{total} треков; "
                f"попыток {s['attempted']}, успешных {s['resolved']}, "
                f"пропущено по размеру лица {s['too_small']}")
