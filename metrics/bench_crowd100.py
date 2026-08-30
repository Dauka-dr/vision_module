"""Does the pipeline hold together at ~100 people, or does it fall apart?

No real 100-person scene is available, so this composites two auditoriums into
one frame. Detection quality on a stitched image is not meaningful, but the
system behaviour at that headcount is: whether the tracker keeps ids stable,
whether per-frame cost stays sane, and whether 100 simultaneous pose buffers
cause trouble.
"""
# Run from anywhere: the project root holds pipeline.py and the test scenes.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import argparse
import time

import cv2
import numpy as np
import torch

from pipeline import PosePipeline


def crowd_frame():
    """One auditorium beside its mirror image — roughly double the people."""
    big = cv2.imread('test_scenes/auditorium_bi.jpg')
    right = big[:, big.shape[1] // 3:]           # the populated half
    return np.hstack([right, cv2.flip(right, 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=15)
    ap.add_argument('--seq-len', type=int, default=20)
    args = ap.parse_args()

    frame = crowd_frame()
    print(f"stitched scene: {frame.shape[1]}x{frame.shape[0]}")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    pipe = PosePipeline(scene='auto', seq_len=args.seq_len)
    rng = np.random.default_rng(0)

    counts, times, ids_seen = [], [], set()
    peak_mem = 0
    for i in range(args.frames):
        noisy = np.clip(frame.astype(np.int16) + rng.integers(-3, 4, frame.shape),
                        0, 255).astype(np.uint8)
        torch.cuda.synchronize()
        t0 = time.time()
        people = pipe.process(noisy)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        counts.append(len(people))
        ids_seen |= {p.track_id for p in people}
        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 2**20)
        if i % 5 == 0:
            ready = sum(p.sequence is not None for p in people)
            print(f"  frame {i:>3}  people {len(people):>3}  tracks "
                  f"{len(pipe.tracker.tracks):>3}  ready {ready:>3}  "
                  f"{times[-1]:.2f}s", flush=True)

    avg = float(np.mean(counts))
    print(f"\n{'people per frame':<26}{avg:.1f}  (±{np.std(counts):.1f})")
    print(f"{'distinct ids':<26}{len(ids_seen)}")
    print(f"{'id churn':<26}{len(ids_seen) / max(avg, 1):.2f}   (1.0 ideal, <2 healthy)")
    print(f"{'median frame time':<26}{np.median(times):.2f}s "
          f"({1 / np.median(times):.1f} FPS)")
    print(f"{'peak GPU memory':<26}{peak_mem:.0f} MB")
    full = sum(1 for t in pipe.tracker.tracks
               if pipe.tracker.sequence(t) is not None)
    print(f"{'full pose sequences':<26}{full}")


if __name__ == '__main__':
    main()
