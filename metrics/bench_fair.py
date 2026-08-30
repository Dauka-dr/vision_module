"""Fair detector comparison: raw counts are inflated by duplicate boxes.

Different RTMDet sizes emit different amounts of overlapping/fragmented boxes, so
comparing raw detection counts flatters the noisier model. This applies one
identical NMS to every model's output before counting, and also reports how many
boxes each model loses to dedup, which is itself a quality signal.
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
from mmengine.config import Config

from bench_detection import build_pipeline, detect, nms, draw

from pipeline import CKPT_DIR as CKPT
MODELS = {
    'rtmdet-t': ('rtmdet_tiny_8xb32-300e_coco.py',
                 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth', 4.8),
    'rtmdet-s': ('rtmdet_s_8xb32-300e_coco.py',
                 'rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth', 8.9),
    'rtmdet-m': ('rtmdet_m_8xb32-300e_coco.py',
                 'rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth', 24.7),
    'rtmdet-l': ('rtmdet_l_8xb32-300e_coco.py',
                 'rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth', 52.3),
    'rtmdet-x': ('rtmdet_x_8xb32-300e_coco.py',
                 'rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth', 94.9),
}


def median_box_height(boxes):
    return float(np.median(boxes[:, 3] - boxes[:, 1])) if len(boxes) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', default='test_scenes/auditorium_bi.jpg')
    ap.add_argument('--scale', type=int, default=1920)
    ap.add_argument('--score-thr', type=float, default=0.25)
    ap.add_argument('--nms-iou', type=float, default=0.5)
    ap.add_argument('--out-dir', default='bench_out')
    args = ap.parse_args()

    from mmdet.apis import init_detector
    import mmdet

    os.makedirs(args.out_dir, exist_ok=True)
    img = cv2.imread(args.img)
    print(f"Scene: {args.img}  input scale {args.scale}, "
          f"score>{args.score_thr}, unified NMS IoU {args.nms_iou}\n")
    print(f"{'model':<11}{'params':>8}{'raw':>7}{'after NMS':>11}"
          f"{'dropped':>9}{'med h(px)':>11}")
    print("-" * 57)

    for name, (cfg_name, ckpt_name, params) in MODELS.items():
        ckpt = f'{CKPT}/{ckpt_name}'
        if not os.path.exists(ckpt):
            print(f"{name:<11}  -- not downloaded --")
            continue
        cfg_path = os.path.join(os.path.dirname(mmdet.__file__),
                                f'.mim/configs/rtmdet/{cfg_name}')
        model = init_detector(cfg_path, ckpt, device='cuda:0')
        pipeline = build_pipeline(Config.fromfile(cfg_path), args.scale)

        raw = detect(model, pipeline, img, args.score_thr)
        deduped = nms(raw, args.nms_iou)
        dropped = len(raw) - len(deduped)
        pct = dropped / max(len(raw), 1) * 100

        print(f"{name:<11}{params:>7.1f}M{len(raw):>7}{len(deduped):>11}"
              f"{f'{dropped} ({pct:.0f}%)':>9}{median_box_height(deduped):>11.0f}")
        cv2.imwrite(f'{args.out_dir}/fair_{name}.jpg',
                    draw(img, deduped, f'{name} @{args.scale} deduped'))
        del model

    print("-" * 57)
    print("\n'dropped' = boxes removed as duplicates of another box on the same "
          "person.\nA high share means the raw count overstated that model's recall.")


if __name__ == '__main__':
    main()
