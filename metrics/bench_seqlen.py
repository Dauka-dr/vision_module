"""How long a pose buffer does the action model actually need?

The model was trained on clips resampled to 100 frames, but the live pipeline
holds a shorter rolling buffer and lets the sampler stretch it. Shorter buffers
mean a person is classified sooner after appearing — at 2-4 fps a 60-frame buffer
is 15-30 seconds of waiting — so the question is what that costs in accuracy.

Also sweeps the confidence threshold, which decides how often the system answers
at all versus how often it is right when it does.
"""
# Run from anywhere: the project root holds pipeline.py and the test scenes.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import argparse
import os
import pickle

import numpy as np

from action_classifier import ActionClassifier, TARGET7


def load_clips(path, n, seed=0):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    val = set(data['split']['xsub_val'])
    clips = [a for a in data['annotations'] if a['frame_dir'] in val]
    np.random.default_rng(seed).shuffle(clips)
    return clips[:n]


def as_sequence(ann, length=None):
    """Full multi-person sequence, optionally cropped to the last `length` frames.

    Cropping from the end mimics a live rolling buffer, which always holds the
    most recent frames.
    """
    kp, sc = ann['keypoint'], ann['keypoint_score']
    seq = np.concatenate([kp, sc[..., None]], axis=-1)
    if length is not None and seq.shape[1] > length:
        seq = seq[:, -length:]
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', default='dataset/annotations/target_7cls.pkl')
    ap.add_argument('--n', type=int, default=400)
    args = ap.parse_args()

    if not os.path.exists(args.ann):
        raise SystemExit(f"нет {args.ann} — соберите его: python build_subset.py")

    clips = load_clips(args.ann, args.n)
    truth = [TARGET7[a['label']] for a in clips]
    shape = tuple(clips[0]['img_shape'])
    clf = ActionClassifier()

    print(f"клипов: {len(clips)}\n")
    print(f"{'длина буфера':>14}{'точность':>11}{'ср. уверенность':>18}")
    print('-' * 45)
    best = None
    preds_by_len = {}
    for length in (20, 30, 45, 60, 80, 100, None):
        seqs = [as_sequence(a, length) for a in clips]
        preds = clf(seqs, img_shape=shape)
        preds_by_len[length] = preds
        acc = np.mean([p[0] == t for p, t in zip(preds, truth)])
        conf = np.mean([p[1] for p in preds])
        label = 'весь клип' if length is None else f'{length} кадров'
        print(f"{label:>14}{acc:>10.1%}{conf:>18.2f}")
        if best is None or acc > best[1]:
            best = (length, acc)
    print('-' * 45)
    print(f"лучшая: {best[0] or 'весь клип'} -> {best[1]:.1%}\n")

    # Threshold sweep on the buffer length the pipeline actually uses.
    preds = preds_by_len[60]
    print("порог уверенности при буфере 60 кадров:")
    print(f"{'порог':>8}{'отвечает':>11}{'точность ответов':>19}")
    print('-' * 40)
    for thr in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        kept = [(p, t) for p, t in zip(preds, truth) if p[1] >= thr]
        rate = len(kept) / len(preds)
        acc = np.mean([p[0] == t for p, t in kept]) if kept else float('nan')
        print(f"{thr:>8.1f}{rate:>10.0%}{acc:>19.1%}")
    print('-' * 40)
    print("«отвечает» — доля людей, для которых система вообще назовёт действие;")
    print("«точность ответов» — насколько верны те ответы, что она даёт.")


if __name__ == '__main__':

    main()
