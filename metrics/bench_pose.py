"""Benchmark: top-down (RTMDet + RTMPose) vs one-stage (RTMO) on a dense scene.

Both produce COCO-17 keypoints, so the comparison is apples to apples.
Reports how many people get a usable skeleton and what it costs per frame.
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
from mmengine.registry import init_default_scope

from pipeline import CKPT_DIR as CKPT, MMPOSE_CFG

RTMO_MODELS = {
    'rtmo-m': (f'{MMPOSE_CFG}/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py',
               f'{CKPT}/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth'),
    'rtmo-l': (f'{MMPOSE_CFG}/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py',
               f'{CKPT}/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth'),
}

SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
            (1, 3), (2, 4), (3, 5), (4, 6)]


def set_scale(cfg, scale):
    """Override the Resize/Pad scale in a pose config's test pipeline."""
    cfg = cfg.copy()
    for t in cfg.test_dataloader.dataset.pipeline:
        if t['type'] == 'Resize':
            t['scale'] = (scale, scale)
        elif t['type'] == 'BottomupResize':
            t['input_size'] = (scale, scale)
    return cfg


def bench(fn, warmup=1, runs=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.time()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    return out, float(np.median(times))


def draw_poses(img, keypoints, scores, label, kpt_thr=0.3):
    vis = img.copy()
    for kpts, sc in zip(keypoints, scores):
        for i, j in SKELETON:
            if sc[i] > kpt_thr and sc[j] > kpt_thr:
                cv2.line(vis, tuple(kpts[i].astype(int)), tuple(kpts[j].astype(int)),
                         (0, 255, 0), 3)
        for k, s in zip(kpts, sc):
            if s > kpt_thr:
                cv2.circle(vis, tuple(k.astype(int)), 4, (0, 0, 255), -1)
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.putText(vis, f"{label}: {len(keypoints)} skeletons", (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
    return cv2.resize(vis, (1600, int(h * 1600 / w)))


def run_rtmo(name, scale, img, score_thr, out_dir):
    from mmpose.apis import init_model, inference_bottomup
    init_default_scope('mmpose')
    cfg_path, ckpt = RTMO_MODELS[name]
    cfg = set_scale(Config.fromfile(cfg_path), scale)
    model = init_model(cfg, ckpt, device='cuda:0')

    def infer():
        res = inference_bottomup(model, img)[0]
        pred = res.pred_instances
        keep = pred.scores > score_thr
        return pred.keypoints[keep], pred.keypoint_scores[keep]

    (kpts, kscores), dt = bench(infer)
    label = f'{name} @{scale}'
    print(f"{label:<34}{len(kpts):>8}{dt:>9.3f}s{1/dt:>8.1f}")
    cv2.imwrite(f'{out_dir}/{name}_{scale}.jpg', draw_poses(img, kpts, kscores, label))
    del model
    torch.cuda.empty_cache()
    return len(kpts), dt


def run_topdown(det_scale, img, score_thr, out_dir):
    """RTMDet-m at det_scale + RTMPose-m on every detected person (batched)."""
    from mmdet.apis import init_detector, inference_detector
    from mmdet.utils import get_test_pipeline_cfg
    from mmpose.apis import init_model, inference_topdown
    import mmdet

    det_cfg_path = os.path.join(os.path.dirname(mmdet.__file__),
                                '.mim/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py')
    det_ckpt = f'{CKPT}/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'
    init_default_scope('mmdet')
    detector = init_detector(det_cfg_path, det_ckpt, device='cuda:0')

    dcfg = Config.fromfile(det_cfg_path)
    pcfg = get_test_pipeline_cfg(dcfg.copy())
    pcfg[0].type = 'mmdet.LoadImageFromNDArray'
    for t in pcfg:
        if t['type'] == 'Resize':
            t['scale'] = (det_scale, det_scale)
        elif t['type'] == 'Pad':
            t['size'] = (det_scale, det_scale)
    det_pipeline = Compose(pcfg)

    init_default_scope('mmpose')
    pose_cfg = (f'{MMPOSE_CFG}/body_2d_keypoint/rtmpose/body8/'
                'rtmpose-m_8xb256-420e_body8-256x192.py')
    pose_ckpt = f'{CKPT}/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'
    pose_model = init_model(pose_cfg, pose_ckpt, device='cuda:0')

    def infer():
        init_default_scope('mmdet')
        res = inference_detector(detector, img, test_pipeline=det_pipeline)
        inst = res.pred_instances.cpu().numpy()
        keep = (inst.labels == 0) & (inst.scores > score_thr)
        bboxes = inst.bboxes[keep]
        if len(bboxes) == 0:
            return np.zeros((0, 17, 2)), np.zeros((0, 17))
        init_default_scope('mmpose')
        pose_results = inference_topdown(pose_model, img, bboxes, bbox_format='xyxy')
        kpts = np.concatenate([r.pred_instances.keypoints for r in pose_results])
        ksc = np.concatenate([r.pred_instances.keypoint_scores for r in pose_results])
        return kpts, ksc

    (kpts, kscores), dt = bench(infer)
    label = f'RTMDet-m@{det_scale} + RTMPose-m'
    print(f"{label:<34}{len(kpts):>8}{dt:>9.3f}s{1/dt:>8.1f}")
    cv2.imwrite(f'{out_dir}/topdown_{det_scale}.jpg', draw_poses(img, kpts, kscores, label))
    del detector, pose_model
    torch.cuda.empty_cache()
    return len(kpts), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', default='test_scenes/auditorium_bi.jpg')
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--out-dir', default='bench_out')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img = cv2.imread(args.img)
    print(f"Scene: {args.img}  {img.shape[1]}x{img.shape[0]}\n")
    print(f"{'config':<34}{'people':>8}{'time':>10}{'FPS':>8}")
    print("-" * 60)

    results = {}
    for det_scale in (640, 1920):
        results[f'topdown@{det_scale}'] = run_topdown(det_scale, img, args.score_thr, args.out_dir)
    for name in ('rtmo-m', 'rtmo-l'):
        if not os.path.exists(RTMO_MODELS[name][1]):
            # RTMO lost this benchmark (38 vs 52 people) and its weights were
            # deleted to save disk. URLs are in BENCHMARKS.md if you want to redo it.
            print(f"{name:<34}  -- нет весов в models/, пропуск --")
            continue
        for scale in (640, 1280, 1920):
            results[f'{name}@{scale}'] = run_rtmo(name, scale, img, args.score_thr, args.out_dir)

    print("-" * 60)
    best = max(results.items(), key=lambda kv: kv[1][0])
    print(f"most skeletons: {best[0]} -> {best[1][0]} people at {best[1][1]:.3f}s/frame")


if __name__ == '__main__':
    main()
