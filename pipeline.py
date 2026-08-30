"""Multi-person pose pipeline: detect -> pose -> track -> per-person pose sequences.

The buffers this produces (T frames x 17 joints x 3) are the input format that
ST-GCN++ / PoseC3D expect, so this is the stage the action classifier plugs into.

Benchmark findings baked into the defaults (see BENCHMARKS.md):
  - Detector input scale matters far more than detector capacity. RTMDet-m at
    1920 found 2x the people of RTMDet-m at 640; RTMDet-l and -x found *fewer*
    while costing more.
  - Pose is batched in a single forward pass over all detected boxes.
"""
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope
from scipy.optimize import linear_sum_assignment

# Weights live inside the project, not in the torch hub cache, so the whole
# thing can be copied to another machine as one folder.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(PROJECT_DIR, 'models')

# Config files ship inside the installed mmpose package.
import mmpose as _mmpose
MMPOSE_CFG = os.path.join(os.path.dirname(_mmpose.__file__), '.mim', 'configs')

DETECTORS = {
    'rtmdet-t': ('rtmdet_tiny_8xb32-300e_coco.py',
                 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
                 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/'
                 'rtmdet_tiny_8xb32-300e_coco/'
                 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'),
    'rtmdet-s': ('rtmdet_s_8xb32-300e_coco.py',
                 'rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth',
                 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/'
                 'rtmdet_s_8xb32-300e_coco/'
                 'rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'),
    'rtmdet-m': ('rtmdet_m_8xb32-300e_coco.py',
                 'rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth',
                 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/'
                 'rtmdet_m_8xb32-300e_coco/'
                 'rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'),
}

POSE_MODELS = {
    'rtmpose-t': (f'{MMPOSE_CFG}/body_2d_keypoint/rtmpose/body8/'
                  'rtmpose-t_8xb256-420e_body8-256x192.py',
                  'rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.pth',
                  'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/'
                  'rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.pth'),
    'rtmpose-s': (f'{MMPOSE_CFG}/body_2d_keypoint/rtmpose/body8/'
                  'rtmpose-s_8xb256-420e_body8-256x192.py',
                  'rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth',
                  'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/'
                  'rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth'),
    'rtmpose-m': (f'{MMPOSE_CFG}/body_2d_keypoint/rtmpose/body8/'
                  'rtmpose-m_8xb256-420e_body8-256x192.py',
                  'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
                  'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/'
                  'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'),
}

# COCO-17 skeleton edges, used for drawing and as the ST-GCN++ graph layout.
COCO_SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                 (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                 (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]

LEFT_WRIST, RIGHT_WRIST = 9, 10

# Objects that disambiguate poses the skeleton cannot separate. "Looking at a
# phone" and "writing in a notebook" put the body in the same posture, so the
# skeleton stream alone cannot tell them apart at any amount of training data —
# what differs is the object in the person's hands. RTMDet already predicts these
# COCO classes on every frame, so this second stream costs no extra inference.
CONTEXT_OBJECTS = {
    67: 'cell_phone',
    73: 'book',
    63: 'laptop',
    66: 'keyboard',
}


def _ensure_checkpoint(filename: str, url: str) -> str:
    """Return local checkpoint path, downloading it on first use."""
    path = os.path.join(CKPT_DIR, filename)
    if not os.path.exists(path):
        os.makedirs(CKPT_DIR, exist_ok=True)
        print(f"Downloading {filename} (first use, this is slow)...", flush=True)
        torch.hub.download_url_to_file(url, path)
    return path


def dedup_nms(boxes: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Drop overlapping duplicate boxes on the same person.

    RTMDet's internal NMS still leaves duplicates in dense scenes, and the
    smaller variants leave a lot of them (39% of rtmdet-t's raw boxes at 1920).
    Without this the tracker spawns several ids per person and each gets its own
    half-filled pose buffer.
    """
    if len(boxes) == 0:
        return boxes
    x1, y1, x2, y2, scores = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return boxes[keep]


class PersonDetector:
    """RTMDet person detector with an explicit, tunable input resolution."""

    def __init__(self, model: str = 'rtmdet-s', input_scale: int = 1600,
                 score_thr: float = 0.25, nms_iou: float = 0.5,
                 object_score_thr: Optional[float] = 0.15,
                 device: str = 'cuda:0'):
        from mmdet.apis import init_detector
        from mmdet.utils import get_test_pipeline_cfg
        import mmdet

        cfg_name, ckpt_name, url = DETECTORS[model]
        cfg_path = os.path.join(os.path.dirname(mmdet.__file__),
                                f'.mim/configs/rtmdet/{cfg_name}')
        init_default_scope('mmdet')
        self.model = init_detector(cfg_path, _ensure_checkpoint(ckpt_name, url),
                                   device=device)
        self.score_thr = score_thr
        self.input_scale = input_scale
        self.nms_iou = nms_iou
        # Objects are much smaller than people, so they need a lower threshold to
        # show up at all — a phone is ~57 px tall where a person is ~300 px.
        self.object_score_thr = object_score_thr

        self._cfg = Config.fromfile(cfg_path)
        self._pipelines = {}

    def _pipeline_for(self, frame_shape) -> Compose:
        """Test pipeline for this frame, never upscaling past its native size.

        Feeding a 720p webcam frame in at 1600 does not add information — it just
        presents people at a scale the detector never saw in training, which makes
        boxes fragment and flicker between frames. Clamping to the native size
        removed the instability entirely (see benchmarks/bench_stability.py).
        """
        from mmdet.utils import get_test_pipeline_cfg
        scale = min(self.input_scale, max(frame_shape[:2]))
        if scale not in self._pipelines:
            pipeline_cfg = get_test_pipeline_cfg(self._cfg.copy())
            pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
            for t in pipeline_cfg:
                if t['type'] == 'Resize':
                    t['scale'] = (scale, scale)
                elif t['type'] == 'Pad':
                    t['size'] = (scale, scale)
            self._pipelines[scale] = Compose(pipeline_cfg)
        return self._pipelines[scale]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Return person boxes as (N, 5): x1, y1, x2, y2, score."""
        return self.detect_all(img)[0]

    def detect_all(self, img: np.ndarray):
        """One forward pass, two streams: people and context objects.

        Returns (person_boxes (N,5), {class_name: (M,5) boxes}). The objects come
        free — the detector predicts all 80 COCO classes anyway, we simply stop
        discarding the ones that matter.
        """
        from mmdet.apis import inference_detector
        init_default_scope('mmdet')
        result = inference_detector(self.model, img,
                                    test_pipeline=self._pipeline_for(img.shape))
        inst = result.pred_instances.cpu().numpy()

        keep = (inst.labels == 0) & (inst.scores > self.score_thr)
        people = dedup_nms(
            np.concatenate([inst.bboxes[keep], inst.scores[keep, None]], axis=1),
            self.nms_iou)

        objects = {}
        if self.object_score_thr is not None:
            for cid, name in CONTEXT_OBJECTS.items():
                keep = (inst.labels == cid) & (inst.scores > self.object_score_thr)
                if keep.sum():
                    objects[name] = dedup_nms(
                        np.concatenate(
                            [inst.bboxes[keep], inst.scores[keep, None]], axis=1),
                        self.nms_iou)
        return people, objects


class PoseEstimator:
    """RTMPose top-down estimator; all boxes go through in one batched forward."""

    def __init__(self, model: str = 'rtmpose-m', device: str = 'cuda:0'):
        from mmpose.apis import init_model
        cfg_path, ckpt_name, url = POSE_MODELS[model]
        init_default_scope('mmpose')
        self.model = init_model(cfg_path, _ensure_checkpoint(ckpt_name, url),
                                device=device)

    def __call__(self, img: np.ndarray, boxes: np.ndarray):
        """Return (keypoints (N,17,2), scores (N,17)) for the given boxes."""
        from mmpose.apis import inference_topdown
        if len(boxes) == 0:
            return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
        init_default_scope('mmpose')
        results = inference_topdown(self.model, img, boxes[:, :4], bbox_format='xyxy')
        kpts = np.concatenate([r.pred_instances.keypoints for r in results])
        scores = np.concatenate([r.pred_instances.keypoint_scores for r in results])
        return kpts, scores


def attach_objects(person_boxes: np.ndarray, keypoints: np.ndarray,
                   kpt_scores: np.ndarray, objects: Dict[str, np.ndarray],
                   reach: float = 0.45, kpt_thr: float = 0.3
                   ) -> List[List[tuple]]:
    """Assign each detected object to the person most plausibly using it.

    A phone lying on a desk is not "someone looking at a phone" — what matters is
    whether it is in reach of that person's hands. So objects are matched to the
    nearest visible wrist, and only within `reach` * person-height of it. Each
    object goes to at most one person (its closest), which matters in dense rows
    where one object sits between two people.

    Returns, per person, a list of (name, det_score, closeness) with closeness in
    0..1 — 1 meaning the object is right at the wrist.
    """
    per_person = [[] for _ in range(len(person_boxes))]
    if len(person_boxes) == 0:
        return per_person

    heights = np.maximum(person_boxes[:, 3] - person_boxes[:, 1], 1.0)

    for name, boxes in objects.items():
        for x1, y1, x2, y2, score in boxes:
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

            best_person, best_dist, best_limit = -1, np.inf, 1.0
            for i in range(len(person_boxes)):
                wrists = [keypoints[i, j] for j in (LEFT_WRIST, RIGHT_WRIST)
                          if kpt_scores[i, j] > kpt_thr]
                if not wrists:
                    continue
                dist = min(np.linalg.norm(center - w) for w in wrists)
                limit = reach * heights[i]
                if dist < limit and dist < best_dist:
                    best_person, best_dist, best_limit = i, dist, limit

            if best_person >= 0:
                closeness = 1.0 - best_dist / best_limit
                per_person[best_person].append((name, float(score), float(closeness)))

    for hits in per_person:
        hits.sort(key=lambda h: -h[2])
    return per_person


def dedup_poses(boxes: np.ndarray, keypoints: np.ndarray, kpt_scores: np.ndarray,
                pose_thr: float = 0.15, kpt_thr: float = 0.3,
                min_joints: int = 4):
    """Drop detections whose skeleton duplicates another's — same person, twice.

    Box-level NMS misses these. Two boxes on one person often have low IoU: a
    head-and-shoulders box against a full-body box overlaps a lot in absolute
    terms but scores poorly on intersection-over-union because the areas differ.
    Raising the box threshold instead is not an option either, since in a packed
    hall one person's box legitimately sits inside another's — the person behind
    them. Measured on the reference scene, 32 pairs were heavily nested and only
    15 of those were actually the same person.

    What separates the two cases is the skeleton: two detections of one person
    put the joints in the same places, two different people never do. Distances
    are normalised by person size so the threshold holds at any distance.

    Returns the indices worth keeping.
    """
    n = len(boxes)
    if n < 2:
        return np.arange(n)

    # Prefer the detection the detector was more sure of, and among equals the
    # one with more visible joints — that is the fuller view of the person.
    quality = boxes[:, 4] + 0.01 * (kpt_scores > kpt_thr).sum(axis=1)
    order = np.argsort(-quality)

    scales = np.sqrt(np.maximum(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1.0))

    keep, dropped = [], np.zeros(n, dtype=bool)
    for i in order:
        if dropped[i]:
            continue
        keep.append(i)
        for j in order:
            if j == i or dropped[j]:
                continue
            both = (kpt_scores[i] > kpt_thr) & (kpt_scores[j] > kpt_thr)
            if both.sum() < min_joints:
                continue
            scale = min(scales[i], scales[j])
            dist = np.linalg.norm(keypoints[i][both] - keypoints[j][both],
                                  axis=1).mean() / scale
            if dist < pose_thr:
                dropped[j] = True
    return np.array(sorted(keep))


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    misses: int = 0
    hits: int = 1
    poses: deque = field(default_factory=deque)
    # Recent frames' object hits, used to suppress one-frame flickers.
    object_history: deque = field(default_factory=lambda: deque(maxlen=10))


def stable_objects(history, min_frames: int = 3) -> List[tuple]:
    """Keep only objects seen on several recent frames for the same person.

    A single-frame detection is usually noise — at a low threshold the detector
    finds and loses a "book" on alternating frames, which is what makes the labels
    flicker. Requiring persistence also makes the evidence more trustworthy for the
    action classifier, which cares whether someone is holding a phone over a span
    of seconds, not in one frame.
    """
    if not history:
        return []
    # A single image has no history to be persistent across, and the first frames
    # of a video have little — require only as many frames as exist so far.
    min_frames = min(min_frames, len(history))
    counts, best = {}, {}
    for frame_hits in history:
        for name, score, closeness in frame_hits:
            counts[name] = counts.get(name, 0) + 1
            if closeness > best.get(name, (0, 0))[1]:
                best[name] = (score, closeness)
    out = [(name, best[name][0], best[name][1])
           for name, n in counts.items() if n >= min_frames]
    out.sort(key=lambda h: -h[2])
    return out


class Tracker:
    """IoU tracker with Hungarian matching — gives each person a stable id.

    Stable ids are what let us accumulate a pose sequence per person, which the
    action classifier consumes. Without tracking there is no temporal signal.
    """

    def __init__(self, iou_thr: float = 0.3, max_age: int = 30,
                 seq_len: int = 100):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.seq_len = seq_len
        self.tracks: Dict[int, Track] = {}
        self._next_id = 0

    @staticmethod
    def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)))
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        lt = np.maximum(a[:, None, :2], b[None, :, :2])
        rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
        wh = np.clip(rb - lt, 0, None)
        inter = wh[..., 0] * wh[..., 1]
        return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)

    def update(self, boxes: np.ndarray, keypoints: np.ndarray,
               kpt_scores: np.ndarray) -> List[int]:
        """Assign ids to this frame's detections; returns one id per box."""
        ids = list(self.tracks.keys())
        track_boxes = np.array([self.tracks[i].box for i in ids]) if ids \
            else np.zeros((0, 4))

        assigned = [-1] * len(boxes)
        matched_tracks = set()

        if len(ids) and len(boxes):
            iou = self._iou_matrix(track_boxes, boxes[:, :4])
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_thr:
                    assigned[c] = ids[r]
                    matched_tracks.add(ids[r])

        for det_idx, track_id in enumerate(assigned):
            if track_id == -1:
                track_id = self._next_id
                self._next_id += 1
                self.tracks[track_id] = Track(
                    track_id, boxes[det_idx, :4],
                    poses=deque(maxlen=self.seq_len))
                assigned[det_idx] = track_id
            else:
                tr = self.tracks[track_id]
                tr.box = boxes[det_idx, :4]
                tr.misses = 0
                tr.hits += 1
            # store (17, 3): x, y, confidence
            pose = np.concatenate(
                [keypoints[det_idx], kpt_scores[det_idx][:, None]], axis=1)
            self.tracks[track_id].poses.append(pose.astype(np.float32))

        for track_id in list(self.tracks):
            if track_id not in set(assigned):
                self.tracks[track_id].misses += 1
                if self.tracks[track_id].misses > self.max_age:
                    del self.tracks[track_id]

        return assigned

    def sequence(self, track_id: int) -> Optional[np.ndarray]:
        """Full pose sequence (T, 17, 3) once enough frames accumulated."""
        tr = self.tracks.get(track_id)
        if tr is None or len(tr.poses) < self.seq_len:
            return None
        return np.stack(tr.poses)


@dataclass
class Person:
    track_id: int
    box: np.ndarray
    keypoints: np.ndarray   # (17, 2)
    kpt_scores: np.ndarray  # (17,)
    sequence: Optional[np.ndarray] = None  # (T, 17, 3) when the buffer is full
    buffer_len: int = 0     # frames of history so far, for progress display
    # Objects within reach of this person's hands: (name, det_score, closeness).
    # This is the evidence the skeleton cannot provide — see attach_objects.
    objects: List[tuple] = field(default_factory=list)


# One set of settings cannot serve both a far, dense auditorium and a close-up
# webcam. The auditorium needs low thresholds and high input resolution to catch
# small distant people; on a close webcam those same settings produce fragmented
# boxes, reshuffling ids and ~30 spurious objects per frame. Numbers behind these
# values: benchmarks/bench_stability.py.
SCENE_PRESETS = {
    'webcam': dict(input_scale=960, score_thr=0.4, object_score_thr=0.4),
    'auditorium': dict(input_scale=1600, score_thr=0.25, object_score_thr=0.15),
}

CANDIDATE_SCALES = (640, 960, 1280, 1600, 1920, 2560, 3200)


def autocalibrate(detector, frames, probe_thr: float = 0.25,
                  verbose: bool = False) -> dict:
    """Pick input_scale and thresholds by trying them on this camera's own frames.

    There is no single "right" person size to aim for: the wide hall shot peaks
    with people around 70 px in the network input (recall on the back rows is what
    matters), while a close-up peaks around 140 px (not fragmenting the big people
    is what matters). A formula tuned to either one hurts the other, so this
    measures instead — it sweeps candidate resolutions on real frames and keeps the
    one that finds the most people without the count wobbling between frames.

    Costs a handful of detector passes once, at startup.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("autocalibrate needs at least one frame")

    # Two near-identical frames are enough to expose instability: with a static
    # scene any disagreement between them is the detector wobbling, not motion.
    if len(frames) == 1:
        rng = np.random.default_rng(0)
        noise = rng.integers(-3, 4, frames[0].shape, dtype=np.int16)
        frames.append(np.clip(frames[0].astype(np.int16) + noise, 0, 255)
                      .astype(np.uint8))

    native = max(frames[0].shape[:2])
    candidates = [s for s in CANDIDATE_SCALES if s <= native] or [native]

    saved_scale, saved_thr = detector.input_scale, detector.score_thr
    detector.score_thr = probe_thr
    results = []
    try:
        for scale in candidates:
            detector.input_scale = scale
            counts = [len(detector(f)) for f in frames]
            mean, spread = float(np.mean(counts)), float(np.std(counts))
            # Prefer more people, but discount a config whose count jumps around —
            # an unstable count is what reshuffles track ids downstream.
            results.append((mean - 2.0 * spread, mean, spread, scale))
    finally:
        detector.input_scale, detector.score_thr = saved_scale, saved_thr

    score, mean, spread, scale = max(results)

    # A crowd needs a permissive threshold so small occluded people survive; a
    # close shot of a few people does not, and pays for permissiveness in flicker.
    crowded = mean >= 8
    score_thr = 0.25 if crowded else 0.4
    object_thr = 0.2 if crowded else 0.4

    if verbose:
        tried = ", ".join(f"{s}:{m:.0f}" for _, m, _, s in results)
        print(f"[autocalibrate] {frames[0].shape[1]}x{frames[0].shape[0]}  "
              f"tried {tried}  -> @{scale}, thr {score_thr} "
              f"({mean:.0f} people, wobble {spread:.1f})", flush=True)

    return dict(input_scale=scale, score_thr=score_thr, object_score_thr=object_thr)


class PosePipeline:
    """Detect -> pose -> track, yielding per-person state ready for the classifier."""

    def __init__(self, detector: str = 'rtmdet-s', pose: str = 'rtmpose-m',
                 scene: str = 'auditorium',
                 input_scale: Optional[int] = None,
                 score_thr: Optional[float] = None,
                 nms_iou: float = 0.5,
                 object_score_thr: Optional[float] = -1.0,
                 object_min_frames: int = 3, pose_nms: Optional[float] = 0.15,
                 seq_len: int = 100, device: str = 'cuda:0'):
        if scene != 'auto' and scene not in SCENE_PRESETS:
            raise ValueError(
                f"scene must be 'auto' or one of {sorted(SCENE_PRESETS)}")
        # 'auto' starts from the wide-shot preset and re-tunes itself on the first
        # frames of the real feed, in calibrate().
        preset = SCENE_PRESETS['auditorium' if scene == 'auto' else scene]
        self.scene = scene
        self.calibrated = scene != 'auto'
        # Explicit arguments win over the preset; -1.0 marks "not given" for
        # object_score_thr, since None legitimately means "no object stream".
        input_scale = preset['input_scale'] if input_scale is None else input_scale
        score_thr = preset['score_thr'] if score_thr is None else score_thr
        if object_score_thr == -1.0:
            object_score_thr = preset['object_score_thr']
        self.detector = PersonDetector(
            model=detector, input_scale=input_scale, score_thr=score_thr,
            nms_iou=nms_iou, object_score_thr=object_score_thr, device=device)
        self.pose = PoseEstimator(pose, device)
        self.tracker = Tracker(seq_len=seq_len)
        # On a single image there is no history, so one frame has to be enough.
        self.object_min_frames = object_min_frames
        # Skeleton-distance threshold for the second dedup pass; None disables it.
        self.pose_nms = pose_nms

    def calibrate(self, frames, verbose: bool = True) -> dict:
        """Tune the detector to this feed using a few of its frames."""
        settings = autocalibrate(self.detector, frames, verbose=verbose)
        self.detector.input_scale = settings['input_scale']
        self.detector.score_thr = settings['score_thr']
        self.detector.object_score_thr = settings['object_score_thr']
        self.calibrated = True
        return settings

    def process(self, frame: np.ndarray) -> List[Person]:
        if not self.calibrated:
            # Calibrating on one frame is enough to size the scene; the tracker
            # has no history yet, so nothing is lost by spending this frame.
            self.calibrate([frame])
        boxes, objects = self.detector.detect_all(frame)
        keypoints, kpt_scores = self.pose(frame, boxes)

        # Second pass of deduplication, now that the skeletons exist: box overlap
        # cannot tell a second detection of one person from the person behind
        # them, and matching skeletons can.
        if self.pose_nms:
            keep = dedup_poses(boxes, keypoints, kpt_scores, self.pose_nms)
            boxes, keypoints, kpt_scores = (boxes[keep], keypoints[keep],
                                            kpt_scores[keep])
        ids = self.tracker.update(boxes, keypoints, kpt_scores)
        held = attach_objects(boxes, keypoints, kpt_scores, objects)

        people = []
        for i, tid in enumerate(ids):
            track = self.tracker.tracks[tid]
            track.object_history.append(held[i])
            people.append(Person(
                track_id=tid, box=boxes[i, :4], keypoints=keypoints[i],
                kpt_scores=kpt_scores[i], sequence=self.tracker.sequence(tid),
                buffer_len=len(track.poses),
                objects=stable_objects(track.object_history,
                                       self.object_min_frames)))
        return people
