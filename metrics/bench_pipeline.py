"""Benchmark the full pipeline: which config gives how many people at what FPS.

This is the deployment-planning table — it measures detect + pose + track end to
end on a real dense auditorium frame, not a synthetic single-person case.
"""
# Run from anywhere: the project root holds pipeline.py and the test scenes.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
import argparse
import os
import time

import cv2
import numpy as np
import torch

from pipeline import PosePipeline

from pipeline import CKPT_DIR as CKPT


def available(pipeline_kwargs):
    """Skip configs whose checkpoints have not been downloaded yet."""
    from pipeline import DETECTORS, POSE_MODELS
    det_ckpt = DETECTORS[pipeline_kwargs['detector']][1]
    pose_ckpt = POSE_MODELS[pipeline_kwargs['pose']][1]
    return (os.path.exists(f'{CKPT}/{det_ckpt}')
            and os.path.exists(f'{CKPT}/{pose_ckpt}'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', default='test_scenes/auditorium_bi.jpg')
    ap.add_argument('--score-thr', type=float, default=0.25)
    ap.add_argument('--runs', type=int, default=3)
    args = ap.parse_args()

    img = cv2.imread(args.img)
    print(f"Scene: {args.img}  {img.shape[1]}x{img.shape[0]}")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    configs = []
    for det in ('rtmdet-t', 'rtmdet-s', 'rtmdet-m'):
        for pose in ('rtmpose-t', 'rtmpose-m'):
            for scale in (640, 1280, 1920):
                configs.append(dict(detector=det, pose=pose, input_scale=scale))

    print(f"{'detector':<11}{'pose':<12}{'scale':>7}{'people':>8}"
          f"{'det':>8}{'pose':>8}{'total':>8}{'FPS':>7}")
    print("-" * 69)

    rows = []
    for cfg in configs:
        if not available(cfg):
            continue
        pipe = PosePipeline(score_thr=args.score_thr, **cfg)

        # warm up, then time detect and pose stages separately
        pipe.process(img)
        det_times, pose_times = [], []
        for _ in range(args.runs):
            torch.cuda.synchronize()
            t0 = time.time()
            boxes = pipe.detector(img)
            torch.cuda.synchronize()
            t1 = time.time()
            pipe.pose(img, boxes)
            torch.cuda.synchronize()
            t2 = time.time()
            det_times.append(t1 - t0)
            pose_times.append(t2 - t1)

        det_t = float(np.median(det_times))
        pose_t = float(np.median(pose_times))
        total = det_t + pose_t
        n = len(boxes)
        rows.append((cfg, n, total))
        print(f"{cfg['detector']:<11}{cfg['pose']:<12}{cfg['input_scale']:>7}{n:>8}"
              f"{det_t:>7.3f}s{pose_t:>7.3f}s{total:>7.3f}s{1/total:>7.1f}")

        del pipe
        torch.cuda.empty_cache()

    print("-" * 69)
    print("\nConfigs reaching real-time-ish rates:")
    for cfg, n, total in sorted(rows, key=lambda r: -r[1]):
        if 1 / total >= 10:
            print(f"  {cfg['detector']:<10}{cfg['pose']:<11}@{cfg['input_scale']:<6}"
                  f"{n:>4} people  {1/total:>5.1f} FPS")


if __name__ == '__main__':
    main()
