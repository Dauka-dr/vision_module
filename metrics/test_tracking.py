"""Smoke test for the tracker: do ids stay stable and do pose buffers fill?

Builds a synthetic pan across the auditorium photo so people move smoothly
between frames, which is what the IoU tracker has to cope with. Without stable
ids there is no per-person pose sequence and no action recognition.
"""
# Run from anywhere: the project root holds pipeline.py and the test scenes.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
import sys

import cv2
import numpy as np

from pipeline import PosePipeline

FRAMES = 40
CROP_W, CROP_H = 2200, 1500
MODE = sys.argv[1] if len(sys.argv) > 1 else 'static'
rng = np.random.default_rng(0)


def main():
    img = cv2.imread('test_scenes/auditorium_bi.jpg')
    h, w = img.shape[:2]
    # The audience sits in the right/lower part of this photo; panning from the
    # origin would crop the empty wall and find nobody.
    x0, y0 = int(w * 0.40), int(h * 0.30)
    span_x = w - CROP_W - x0
    span_y = h - CROP_H - y0

    pipe = PosePipeline(seq_len=20)

    id_history = []
    print(f"mode: {MODE} camera\n")
    print(f"{'frame':>6}{'people':>8}{'tracks':>8}{'new ids':>9}{'ready':>7}")
    print("-" * 38)

    seen = set()
    for i in range(FRAMES):
        if MODE == 'pan':
            # camera pans: people genuinely enter and leave, so new ids are expected
            t = i / max(FRAMES - 1, 1)
            x = x0 + int(span_x * t)
            y = y0 + int(span_y * t)
        else:
            # fixed camera with a few px of jitter — the actual deployment case
            x = x0 + span_x // 2 + int(rng.integers(-4, 5))
            y = y0 + span_y // 2 + int(rng.integers(-4, 5))
        frame = img[y:y + CROP_H, x:x + CROP_W]

        people = pipe.process(frame)
        ids = {p.track_id for p in people}
        new = len(ids - seen)
        seen |= ids
        ready = sum(p.sequence is not None for p in people)
        id_history.append(ids)

        if i % 5 == 0 or i == FRAMES - 1:
            print(f"{i:>6}{len(people):>8}{len(pipe.tracker.tracks):>8}"
                  f"{new:>9}{ready:>7}")

    print("-" * 38)
    total_ids = len(seen)
    avg_people = np.mean([len(s) for s in id_history])
    # A tracker that constantly loses people would mint far more ids than there
    # are people on screen.
    churn = total_ids / max(avg_people, 1)
    print(f"avg people per frame : {avg_people:.1f}")
    print(f"distinct ids created : {total_ids}")
    print(f"id churn ratio       : {churn:.2f}  (1.0 = perfect, <2 is healthy)")

    full = [tid for tid in pipe.tracker.tracks
            if pipe.tracker.sequence(tid) is not None]
    print(f"tracks with a full {pipe.tracker.seq_len}-frame sequence: {len(full)}")
    if full:
        seq = pipe.tracker.sequence(full[0])
        print(f"example sequence shape: {seq.shape}  (T, joints, x/y/conf)")
        print("this is exactly the tensor layout ST-GCN++ / PoseC3D consume")


if __name__ == '__main__':
    main()
