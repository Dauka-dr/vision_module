"""Benchmark: how many people RTMDet finds in a dense auditorium scene.

Compares plain inference at several input resolutions against tiled (SAHI-style)
inference, to answer whether the bottleneck is model capacity or input pixels.
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
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import get_test_pipeline_cfg

PERSON_CAT_ID = 0  # COCO 'person'


def build_pipeline(cfg, scale):
    """Test pipeline with a custom Resize/Pad scale, for ndarray input."""
    cfg = cfg.copy()
    pipeline_cfg = get_test_pipeline_cfg(cfg)
    pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    for t in pipeline_cfg:
        if t['type'] == 'Resize':
            t['scale'] = (scale, scale)
        elif t['type'] == 'Pad':
            t['size'] = (scale, scale)
    return Compose(pipeline_cfg)


def detect(model, pipeline, img, score_thr):
    """Run detector, return person boxes as (N, 5) [x1,y1,x2,y2,score]."""
    result = inference_detector(model, img, test_pipeline=pipeline)
    inst = result.pred_instances.cpu().numpy()
    keep = (inst.labels == PERSON_CAT_ID) & (inst.scores > score_thr)
    return np.concatenate([inst.bboxes[keep], inst.scores[keep, None]], axis=1)


def nms(boxes, iou_thr=0.55):
    """Plain IoU NMS over (N, 5) boxes."""
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


def detect_tiled(model, pipeline, img, score_thr, rows, cols, overlap=0.2):
    """Slice the frame into overlapping tiles, detect in each, merge with NMS."""
    h, w = img.shape[:2]
    tile_h, tile_w = h / rows, w / cols
    pad_h, pad_w = tile_h * overlap, tile_w * overlap

    all_boxes = []
    for r in range(rows):
        for c in range(cols):
            y1 = max(int(r * tile_h - pad_h), 0)
            y2 = min(int((r + 1) * tile_h + pad_h), h)
            x1 = max(int(c * tile_w - pad_w), 0)
            x2 = min(int((c + 1) * tile_w + pad_w), w)
            boxes = detect(model, pipeline, img[y1:y2, x1:x2], score_thr)
            if len(boxes):
                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1
                all_boxes.append(boxes)

    if not all_boxes:
        return np.zeros((0, 5))
    return nms(np.concatenate(all_boxes))


def draw(img, boxes, label):
    vis = img.copy()
    for x1, y1, x2, y2, score in boxes:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.putText(vis, f"{label}: {len(boxes)} people", (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
    return cv2.resize(vis, (1600, int(h * 1600 / w)))


def bench(fn, warmup=1, runs=3):
    """Time a callable, return (result, median_seconds)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.time()
        result = fn()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    return result, float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', default='test_scenes/auditorium_bi.jpg')
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--out-dir', default='bench_out')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img = cv2.imread(args.img)
    h, w = img.shape[:2]
    print(f"Scene: {args.img}  {w}x{h}\n")

    import mmdet
    cfg_path = os.path.join(os.path.dirname(mmdet.__file__),
                            '.mim/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py')
    from pipeline import CKPT_DIR
    ckpt = os.path.join(
        CKPT_DIR, 'rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth')
    model = init_detector(cfg_path, ckpt, device='cuda:0')
    cfg = Config.fromfile(cfg_path)

    print(f"{'config':<34}{'people':>8}{'time':>10}{'FPS':>8}")
    print("-" * 60)
    results = {}

    # Plain inference at increasing input resolution
    for scale in (640, 1280, 1920, 2560):
        pipeline = build_pipeline(cfg, scale)
        boxes, dt = bench(lambda: detect(model, pipeline, img, args.score_thr))
        results[f'plain_{scale}'] = (boxes, dt)
        print(f"{f'RTMDet-m @ {scale}x{scale}':<34}{len(boxes):>8}{dt:>9.3f}s{1/dt:>8.1f}")
        cv2.imwrite(f'{args.out_dir}/plain_{scale}.jpg',
                    draw(img, boxes, f'RTMDet-m @{scale}'))

    # Tiled inference — each tile processed at native 640
    pipeline_640 = build_pipeline(cfg, 640)
    for rows, cols in ((2, 3), (3, 4)):
        boxes, dt = bench(
            lambda: detect_tiled(model, pipeline_640, img, args.score_thr, rows, cols),
            warmup=0, runs=2)
        key = f'tiled_{rows}x{cols}'
        results[key] = (boxes, dt)
        print(f"{f'RTMDet-m tiled {rows}x{cols} @640':<34}{len(boxes):>8}{dt:>9.3f}s{1/dt:>8.1f}")
        cv2.imwrite(f'{args.out_dir}/{key}.jpg',
                    draw(img, boxes, f'Tiled {rows}x{cols}'))

    best = max(results.items(), key=lambda kv: len(kv[1][0]))
    baseline = len(results['plain_640'][0])
    print("-" * 60)
    print(f"baseline (default 640): {baseline} people")
    print(f"best: {best[0]} with {len(best[1][0])} people "
          f"({len(best[1][0]) / max(baseline, 1):.1f}x more)")
    print(f"\nAnnotated outputs written to {args.out_dir}/")


if __name__ == '__main__':
    main()
