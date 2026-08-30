"""Benchmark: does a bigger detector beat a higher input resolution?

Runs RTMDet-m / -l / -x over the same dense scene at several input scales, so the
two levers (model capacity vs input pixels) can be compared directly.
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
import numpy as np
import torch
from mmengine.config import Config

from bench_detection import build_pipeline, detect, draw, bench

from pipeline import CKPT_DIR as CKPT
MODELS = {
    'rtmdet-m': ('rtmdet_m_8xb32-300e_coco.py',
                 'rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth', 24.7),
    'rtmdet-l': ('rtmdet_l_8xb32-300e_coco.py',
                 'rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth', 52.3),
    'rtmdet-x': ('rtmdet_x_8xb32-300e_coco.py',
                 'rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth', 94.9),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', default='test_scenes/auditorium_bi.jpg')
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--out-dir', default='bench_out')
    ap.add_argument('--scales', type=int, nargs='+', default=[640, 1280, 1920])
    args = ap.parse_args()

    from mmdet.apis import init_detector
    import mmdet

    os.makedirs(args.out_dir, exist_ok=True)
    img = cv2.imread(args.img)
    print(f"Scene: {args.img}  {img.shape[1]}x{img.shape[0]}\n")
    print(f"{'model':<12}{'params':>9}{'scale':>8}{'people':>8}{'time':>10}{'FPS':>7}")
    print("-" * 55)

    table = {}
    for name, (cfg_name, ckpt_name, params) in MODELS.items():
        ckpt = f'{CKPT}/{ckpt_name}'
        if not os.path.exists(ckpt):
            print(f"{name:<12}  -- checkpoint not downloaded, skipped --")
            continue
        cfg_path = os.path.join(os.path.dirname(mmdet.__file__),
                                f'.mim/configs/rtmdet/{cfg_name}')
        model = init_detector(cfg_path, ckpt, device='cuda:0')
        cfg = Config.fromfile(cfg_path)

        for scale in args.scales:
            pipeline = build_pipeline(cfg, scale)
            boxes, dt = bench(lambda: detect(model, pipeline, img, args.score_thr))
            table[(name, scale)] = (len(boxes), dt)
            print(f"{name:<12}{params:>8.1f}M{scale:>8}{len(boxes):>8}"
                  f"{dt:>9.3f}s{1/dt:>7.1f}")
            cv2.imwrite(f'{args.out_dir}/{name}_{scale}.jpg',
                        draw(img, boxes, f'{name} @{scale}'))

        del model
        torch.cuda.empty_cache()

    print("-" * 55)
    print("\nLever comparison (people found):")
    base = table.get(('rtmdet-m', 640))
    if base:
        print(f"  baseline  rtmdet-m @640       : {base[0]:>3}  ({base[1]:.3f}s)")
        for key, tag in ((('rtmdet-l', 640), 'bigger model, same pixels'),
                         (('rtmdet-x', 640), 'biggest model, same pixels'),
                         (('rtmdet-m', 1920), 'same model, more pixels')):
            if key in table:
                n, dt = table[key]
                delta = (n - base[0]) / max(base[0], 1) * 100
                print(f"  {tag:<26}: {n:>3}  ({dt:.3f}s)  {delta:+.0f}%")


if __name__ == '__main__':
    main()
