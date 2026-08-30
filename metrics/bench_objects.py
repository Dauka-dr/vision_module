"""Can the detector actually see phones and books in a lecture hall?

The pose stream cannot separate "looking at phone" from "writing in a notebook" —
the body does the same thing. The proposed fix is a second stream using the object
classes RTMDet already predicts. But a phone at lecture-hall distance is tiny, and
COCO detectors are weak on small objects, so this measures whether the evidence is
actually there before anything is built on top of it.
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

from bench_detection import build_pipeline, nms
from pipeline import CKPT_DIR

# COCO ids for the objects that disambiguate the confusable classes
OBJECTS = {
    67: ('cell_phone', (0, 0, 255)),
    73: ('book', (0, 255, 255)),
    63: ('laptop', (255, 0, 0)),
    66: ('keyboard', (255, 0, 255)),
    62: ('tv', (128, 128, 0)),
    41: ('cup', (0, 128, 255)),
}


def detect_all(model, pipeline, img, score_thr):
    """Return {class_id: (N,5) boxes} for every class we care about, plus people."""
    from mmdet.apis import inference_detector
    result = inference_detector(model, img, test_pipeline=pipeline)
    inst = result.pred_instances.cpu().numpy()
    out = {}
    for cid in list(OBJECTS) + [0]:
        keep = (inst.labels == cid) & (inst.scores > score_thr)
        if keep.sum():
            out[cid] = nms(
                np.concatenate([inst.bboxes[keep], inst.scores[keep, None]], axis=1),
                0.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', default='test_scenes/auditorium_bi.jpg')
    ap.add_argument('--detector', default='rtmdet-m')
    ap.add_argument('--out-dir', default='bench_out')
    args = ap.parse_args()

    from mmdet.apis import init_detector
    from pipeline import DETECTORS
    import mmdet

    os.makedirs(args.out_dir, exist_ok=True)
    img = cv2.imread(args.img)
    cfg_name, ckpt_name, _ = DETECTORS[args.detector]
    cfg_path = os.path.join(os.path.dirname(mmdet.__file__),
                            f'.mim/configs/rtmdet/{cfg_name}')
    model = init_detector(cfg_path, os.path.join(CKPT_DIR, ckpt_name), device='cuda:0')
    cfg = Config.fromfile(cfg_path)

    print(f"Scene: {args.img}  {img.shape[1]}x{img.shape[0]}  detector {args.detector}\n")

    # Objects are far smaller than people, so sweep resolution and a low threshold.
    for score_thr in (0.3, 0.15):
        print(f"--- score > {score_thr} ---")
        header = f"{'scale':>7}{'person':>8}" + "".join(
            f"{n:>11}" for _, (n, _) in OBJECTS.items())
        print(header)
        print("-" * len(header))
        for scale in (1280, 1920, 2560):
            pipeline = build_pipeline(cfg, scale)
            found = detect_all(model, pipeline, img, score_thr)
            row = f"{scale:>7}{len(found.get(0, [])):>8}"
            for cid, (name, _) in OBJECTS.items():
                row += f"{len(found.get(cid, [])):>11}"
            print(row)
        print()

    # Draw the best case so the detections can be eyeballed
    pipeline = build_pipeline(cfg, 1920)
    found = detect_all(model, pipeline, img, 0.15)
    vis = img.copy()
    for cid, (name, color) in OBJECTS.items():
        for x1, y1, x2, y2, sc in found.get(cid, []):
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            cv2.putText(vis, f"{name} {sc:.2f}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    h, w = vis.shape[:2]
    out = f'{args.out_dir}/objects.jpg'
    cv2.imwrite(out, cv2.resize(vis, (2000, int(h * 2000 / w))))
    print(f"Annotated objects written to {out}")

    # Size statistics — the crux: if phones are a few px, the evidence is unusable
    print(f"\n{'object':<12}{'n':>4}{'median h(px)':>14}{'min':>7}{'max':>7}")
    print("-" * 44)
    for cid, (name, _) in OBJECTS.items():
        boxes = found.get(cid, np.zeros((0, 5)))
        if len(boxes) == 0:
            print(f"{name:<12}{0:>4}{'—':>14}")
            continue
        hs = boxes[:, 3] - boxes[:, 1]
        print(f"{name:<12}{len(boxes):>4}{np.median(hs):>14.0f}{hs.min():>7.0f}{hs.max():>7.0f}")


if __name__ == '__main__':
    main()
