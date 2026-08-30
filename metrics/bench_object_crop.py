"""Does crop-and-upscale rescue phone detection, the way it rescued pose?

Top-down pose beats one-stage because it crops each person and upscales the crop,
recovering detail the downscaled full frame threw away. A phone is far smaller
than a person, so it should benefit from the same trick even more: at 1920 input a
phone is ~57 px, but in a native-resolution crop around one person it is hundreds.

Compares detecting objects on the whole (downscaled) frame against detecting them
inside per-person crops taken at native resolution.
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
from mmengine.config import Config

from bench_detection import build_pipeline, nms
from pipeline import CKPT_DIR, DETECTORS, CONTEXT_OBJECTS, PosePipeline


def detect_objects(model, pipeline, img, score_thr):
    """Return {name: (N,5)} for context objects in this image."""
    from mmdet.apis import inference_detector
    result = inference_detector(model, img, test_pipeline=pipeline)
    inst = result.pred_instances.cpu().numpy()
    out = {}
    for cid, name in CONTEXT_OBJECTS.items():
        keep = (inst.labels == cid) & (inst.scores > score_thr)
        if keep.sum():
            out[name] = nms(np.concatenate(
                [inst.bboxes[keep], inst.scores[keep, None]], axis=1), 0.5)
    return out


def crop_around(img, box, pad=0.6):
    """Region around a person, padded — the desk in front holds the objects."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box[:4]
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(int(x1 - bw * pad), 0)
    cy1 = max(int(y1 - bh * pad * 0.3), 0)
    cx2 = min(int(x2 + bw * pad), w)
    cy2 = min(int(y2 + bh * pad), h)
    return img[cy1:cy2, cx1:cx2], (cx1, cy1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', default='test_scenes/auditorium_bi.jpg')
    ap.add_argument('--detector', default='rtmdet-s')
    ap.add_argument('--score-thr', type=float, default=0.15)
    ap.add_argument('--crop-scale', type=int, default=640,
                    help="detector input size for each crop")
    ap.add_argument('--out-dir', default='bench_out')
    args = ap.parse_args()

    from mmdet.apis import init_detector
    import mmdet

    os.makedirs(args.out_dir, exist_ok=True)
    img = cv2.imread(args.img)
    cfg_name, ckpt_name, _ = DETECTORS[args.detector]
    cfg_path = os.path.join(os.path.dirname(mmdet.__file__),
                            f'.mim/configs/rtmdet/{cfg_name}')
    model = init_detector(cfg_path, os.path.join(CKPT_DIR, ckpt_name), device='cuda:0')
    cfg = Config.fromfile(cfg_path)

    print(f"Scene: {args.img}  {img.shape[1]}x{img.shape[0]}\n")

    # people, so we know where to crop
    pipe = PosePipeline(detector=args.detector, input_scale=1600)
    people = pipe.process(img)
    print(f"people: {len(people)}\n")

    # PosePipeline leaves the default scope on mmpose; building an mmdet test
    # pipeline needs it switched back.
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet')

    print(f"{'approach':<34}" + "".join(f"{n:>12}" for n in CONTEXT_OBJECTS.values())
          + f"{'time':>9}")
    print("-" * 95)

    # --- baseline: objects on the whole downscaled frame ---
    baseline_phones = np.zeros((0, 5))
    for scale in (1920, 2560):
        pipeline = build_pipeline(cfg, scale)
        t0 = time.time()
        found = detect_objects(model, pipeline, img, args.score_thr)
        dt = time.time() - t0
        if scale == 1920:
            baseline_phones = found.get('cell_phone', np.zeros((0, 5)))
        row = f"{f'whole frame @{scale}':<34}"
        row += "".join(f"{len(found.get(n, [])):>12}" for n in CONTEXT_OBJECTS.values())
        print(row + f"{dt:>8.2f}s")

    # --- crops at native resolution, one detector pass per person ---
    pipeline = build_pipeline(cfg, args.crop_scale)
    t0 = time.time()
    merged = {name: [] for name in CONTEXT_OBJECTS.values()}
    crop_sizes = []
    for p in people:
        crop, (ox, oy) = crop_around(img, p.box)
        if crop.size == 0:
            continue
        crop_sizes.append(crop.shape[0])
        found = detect_objects(model, pipeline, crop, args.score_thr)
        for name, boxes in found.items():
            boxes = boxes.copy()
            boxes[:, [0, 2]] += ox
            boxes[:, [1, 3]] += oy
            merged[name].append(boxes)
    dt = time.time() - t0

    merged = {n: (nms(np.concatenate(b), 0.5) if b else np.zeros((0, 5)))
              for n, b in merged.items()}
    row = f"{f'per-person crops @{args.crop_scale}':<34}"
    row += "".join(f"{len(merged[n]):>12}" for n in CONTEXT_OBJECTS.values())
    print(row + f"{dt:>8.2f}s")
    print("-" * 95)
    print(f"\nmedian crop height: {np.median(crop_sizes):.0f} px "
          f"(vs {img.shape[0]} px full frame — that is the upscale factor)")

    # phone sizes, the whole point of the exercise
    for label, boxes in (('whole frame @1920', baseline_phones),
                         ('per-person crops', merged['cell_phone'])):
        if len(boxes):
            hs = boxes[:, 3] - boxes[:, 1]
            print(f"  cell_phone via {label:<20} n={len(boxes):<3} "
                  f"median height {np.median(hs):.0f} px")
        else:
            print(f"  cell_phone via {label:<20} none")

    vis = img.copy()
    for name, boxes in merged.items():
        for x1, y1, x2, y2, sc in boxes:
            color = (0, 0, 255) if name == 'cell_phone' else (0, 200, 200)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            cv2.putText(vis, f"{name} {sc:.2f}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    h, w = vis.shape[:2]
    cv2.imwrite(f'{args.out_dir}/objects_cropped.jpg',
                cv2.resize(vis, (2000, int(h * 2000 / w))))
    print(f"\nAnnotated: {args.out_dir}/objects_cropped.jpg")


if __name__ == '__main__':
    main()
