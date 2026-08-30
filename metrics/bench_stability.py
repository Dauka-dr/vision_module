"""Measures the "chaos": how unstable the output is between near-identical frames.

Symptoms reported on webcam — ids reshuffling, extra boxes on one person,
flickering object labels, jittering joints — are all the same underlying thing:
the detector gives a different answer on frames that barely differ. This puts a
number on it so a fix can be shown to work rather than asserted.
"""
# Run from anywhere: the project root holds pipeline.py and the test scenes.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import argparse

import cv2
import numpy as np

from pipeline import PosePipeline


def webcam_like_frames(n=12, size=(1280, 720), seed=0):
    """A static scene shot by a handheld-ish camera: same content, tiny noise.

    Any variation in the output across these frames is instability, not motion.
    """
    big = cv2.imread('test_scenes/auditorium_bi.jpg')
    crop = big[2400:3400, 3600:5400]
    base = cv2.resize(crop, size)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        noise = rng.integers(-3, 4, base.shape, dtype=np.int16)
        yield np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def measure(pipe, frames):
    counts, obj_counts, ids_seen, per_frame_ids = [], [], set(), []
    joint_positions = {}

    for frame in frames:
        people = pipe.process(frame)
        counts.append(len(people))
        obj_counts.append(sum(len(p.objects) for p in people))
        ids = {p.track_id for p in people}
        per_frame_ids.append(ids)
        ids_seen |= ids
        for p in people:
            joint_positions.setdefault(p.track_id, []).append(p.keypoints.copy())

    avg_people = float(np.mean(counts)) if counts else 0.0
    churn = len(ids_seen) / max(avg_people, 1)

    # joint jitter: how far joints move between frames for a stable track,
    # normalised by that person's box height (scene is static, so this is noise)
    jitters = []
    for tid, seq in joint_positions.items():
        if len(seq) < 5:
            continue
        arr = np.stack(seq)
        step = np.linalg.norm(np.diff(arr, axis=0), axis=-1)
        jitters.append(np.median(step))

    return dict(
        people=avg_people,
        people_std=float(np.std(counts)),
        objects=float(np.mean(obj_counts)),
        objects_std=float(np.std(obj_counts)),
        ids=len(ids_seen),
        churn=churn,
        jitter=float(np.median(jitters)) if jitters else float('nan'),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=12)
    args = ap.parse_args()

    configs = [
        ("сейчас (аудиторные дефолты)", dict(input_scale=1600, score_thr=0.25,
                                             object_score_thr=0.15)),
        ("нативное разрешение кадра", dict(input_scale=1280, score_thr=0.25,
                                           object_score_thr=0.15)),
        ("+ порог людей 0.4", dict(input_scale=1280, score_thr=0.4,
                                   object_score_thr=0.15)),
        ("+ порог объектов 0.4", dict(input_scale=1280, score_thr=0.4,
                                      object_score_thr=0.4)),
        ("+ вход 960", dict(input_scale=960, score_thr=0.4,
                            object_score_thr=0.4)),
    ]

    print(f"{'конфигурация':<30}{'людей':>8}{'±':>6}{'объектов':>10}{'±':>6}"
          f"{'ID':>5}{'churn':>7}{'дрожь':>8}")
    print("-" * 80)
    for label, kwargs in configs:
        pipe = PosePipeline(**kwargs)
        m = measure(pipe, webcam_like_frames(args.frames))
        print(f"{label:<30}{m['people']:>8.1f}{m['people_std']:>6.1f}"
              f"{m['objects']:>10.1f}{m['objects_std']:>6.1f}"
              f"{m['ids']:>5}{m['churn']:>7.2f}{m['jitter']:>8.1f}")
        del pipe
    print("-" * 80)
    print("людей/объектов ± — разброс между почти одинаковыми кадрами (меньше лучше)")
    print("churn — сколько ID заведено на одного человека (1.0 идеал)")
    print("дрожь — медианное смещение сустава между кадрами, px (сцена статична)")


if __name__ == '__main__':
    main()
