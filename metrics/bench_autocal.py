"""Does autocalibration hold up across cameras and resolutions?

The same hall shot by a 4K camera, a 1080p one and a 720p one puts people at
wildly different pixel sizes. A fixed input_scale can only be right for one of
them. This feeds the identical scene at several resolutions and checks that
autocalibration lands on settings that keep recall — and that a close-up webcam
frame does not get treated like a hall.
"""
# Run from anywhere: the project root holds pipeline.py and the test scenes.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import cv2
import numpy as np

from pipeline import PosePipeline, SCENE_PRESETS


def variants():
    """The same auditorium as several cameras, plus a close-up feed."""
    big = cv2.imread('test_scenes/auditorium_bi.jpg')
    for label, width in (('зал 5472px (исходник)', 5472),
                         ('зал 4K 3840px', 3840),
                         ('зал 1080p', 1920),
                         ('зал 720p', 1280)):
        h = int(big.shape[0] * width / big.shape[1])
        yield label, cv2.resize(big, (width, h))

    crop = big[2400:3400, 3600:5400]
    yield 'крупный план 720p', cv2.resize(crop, (1280, 720))


def run(frame, **kwargs):
    pipe = PosePipeline(**kwargs)
    people = pipe.process(frame)
    # second pass on a jittered copy, to see whether the count is stable
    rng = np.random.default_rng(0)
    noisy = np.clip(frame.astype(np.int16) + rng.integers(-3, 4, frame.shape),
                    0, 255).astype(np.uint8)
    people2 = pipe.process(noisy)
    settings = (pipe.detector.input_scale, pipe.detector.score_thr)
    del pipe
    return len(people), abs(len(people) - len(people2)), settings


def main():
    print(f"{'сцена':<24}{'фикс 1600':>11}{'auto':>7}{'подобрал':>16}{'дрейф':>7}")
    print("-" * 68)
    for label, frame in variants():
        fixed, _, _ = run(frame, scene='auditorium')
        auto, drift, (scale, thr) = run(frame, scene='auto')
        gain = f"{auto - fixed:+d}" if auto != fixed else "="
        print(f"{label:<24}{fixed:>11}{auto:>7}"
              f"{f'@{scale} thr{thr}':>16}{drift:>7}  {gain}")
    print("-" * 68)
    print("фикс 1600 — прежнее поведение с одним input_scale на все сцены")
    print("дрейф — расхождение счёта между двумя почти одинаковыми кадрами")


if __name__ == '__main__':
    main()
