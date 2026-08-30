"""Is RTMDet-l's lower recall real, or just a stricter score calibration?

Sweeps the confidence threshold for -m and -l at the same input scale. If the
gap closes at low thresholds, it was calibration; if it persists, -l genuinely
finds fewer people in this scene.
"""
# Run from anywhere: the project root holds pipeline.py and the test scenes.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
import argparse
import os

import cv2
from mmengine.config import Config

from bench_detection import build_pipeline, detect

from pipeline import CKPT_DIR as CKPT
MODELS = {
    'rtmdet-m': ('rtmdet_m_8xb32-300e_coco.py',
                 'rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'),
    'rtmdet-l': ('rtmdet_l_8xb32-300e_coco.py',
                 'rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', default='test_scenes/auditorium_bi.jpg')
    ap.add_argument('--scale', type=int, default=1920)
    args = ap.parse_args()

    from mmdet.apis import init_detector
    import mmdet

    img = cv2.imread(args.img)
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]

    print(f"Scene: {args.img}  input scale {args.scale}\n")
    print(f"{'model':<12}" + "".join(f"{f'thr={t}':>10}" for t in thresholds))
    print("-" * (12 + 10 * len(thresholds)))

    counts = {}
    for name, (cfg_name, ckpt_name) in MODELS.items():
        ckpt = f'{CKPT}/{ckpt_name}'
        if not os.path.exists(ckpt):
            # rtmdet-l lost the benchmark and was deleted to save disk; see
            # MODELS above for the filename and BENCHMARKS.md for the URL.
            print(f"{name:<12}  -- нет весов в models/, пропуск --")
            continue
        cfg_path = os.path.join(os.path.dirname(mmdet.__file__),
                                f'.mim/configs/rtmdet/{cfg_name}')
        model = init_detector(cfg_path, ckpt, device='cuda:0')
        pipeline = build_pipeline(Config.fromfile(cfg_path), args.scale)
        # Detect once at the lowest threshold, then filter — same forward pass.
        boxes = detect(model, pipeline, img, min(thresholds))
        row = [int((boxes[:, 4] > t).sum()) for t in thresholds]
        counts[name] = row
        print(f"{name:<12}" + "".join(f"{n:>10}" for n in row))
        del model

    print("-" * (12 + 10 * len(thresholds)))
    if 'rtmdet-m' in counts and 'rtmdet-l' in counts:
        print("\ngap (m - l) at each threshold:")
        for t, m, l in zip(thresholds, counts['rtmdet-m'], counts['rtmdet-l']):
            print(f"  thr={t:<5} m={m:<4} l={l:<4} diff={m - l:+d}")


if __name__ == '__main__':
    main()
