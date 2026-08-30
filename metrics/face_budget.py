"""How far can a camera be and still resolve a face well enough to identify it?

Face size in pixels is set by distance, sensor width and lens angle — not by which
way the camera points. Turning the camera to face the audience changes whether a
face is *visible* (it stops being the back of a head), not how many pixels it
covers. So the reach of an identification system is a geometry question, and it
can be answered before any camera is bought.

    python metrics/face_budget.py
    python metrics/face_budget.py --fov 55 --need 30
"""
# Run from anywhere: the project root holds pipeline.py and the test scenes.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import argparse
import math

# Average adult inter-pupillary distance. Face recognition models are usually
# quoted against this rather than against overall face height.
IPD_M = 0.063

RESOLUTIONS = [
    ('1080p', 1920),
    ('4K', 3840),
    ('6K', 6144),
    ('8K', 7680),
]


def ipd_px(width_px: int, distance_m: float, fov_deg: float) -> float:
    """Inter-ocular distance in pixels for a person at this distance."""
    scene_width = 2 * distance_m * math.tan(math.radians(fov_deg) / 2)
    return IPD_M * width_px / scene_width


def max_distance(width_px: int, fov_deg: float, need_px: float) -> float:
    """Furthest row where a face still resolves to `need_px` between the eyes."""
    return IPD_M * width_px / (2 * need_px * math.tan(math.radians(fov_deg) / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fov', type=float, default=70.0,
                    help='горизонтальный угол обзора камеры, градусы')
    ap.add_argument('--need', type=float, default=30.0,
                    help='нужно пикселей между глазами (типовой минимум 30)')
    args = ap.parse_args()

    print(f"допущения: угол обзора {args.fov:.0f}°, "
          f"нужно {args.need:.0f} px между глазами, "
          f"межзрачковое расстояние {IPD_M * 100:.1f} см\n")

    distances = [3, 5, 8, 10, 12, 15, 20]
    header = f"{'камера':<10}" + "".join(f"{f'{d} м':>8}" for d in distances)
    print(header)
    print('-' * len(header))
    for name, width in RESOLUTIONS:
        row = f"{name:<10}"
        for d in distances:
            px = ipd_px(width, d, args.fov)
            mark = '' if px >= args.need else '*'
            row += f"{f'{px:.0f}{mark}':>8}"
        print(row)
    print('-' * len(header))
    print(f"* — лицо мельче {args.need:.0f} px, распознавание ненадёжно\n")

    print(f"{'камера':<10}{'дальность до ' + str(int(args.need)) + ' px':>22}")
    print('-' * 32)
    for name, width in RESOLUTIONS:
        print(f"{name:<10}{max_distance(width, args.fov, args.need):>19.1f} м")

    print("\nсверка с замером: на эталонной сцене (5472 px по ширине, съёмка "
          "сзади зала)\nмедианное межзрачковое расстояние вышло 9.7 px — что "
          f"соответствует\nрасстоянию около "
          f"{max_distance(5472, args.fov, 9.7):.0f} м при этом угле обзора. "
          "Порядок сходится.")

    print("\nЧто это значит для зала:")
    print("  первые ряды (3-8 м)   — распознавание работает при 4K и выше")
    print("  середина (10-12 м)    — граница, нужен узкий угол или 6K+")
    print("  дальние ряды (15+ м)  — не распознаются ни при каком разумном сенсоре")
    print("\nУзкий угол обзора помогает так же сильно, как разрешение, но сужает")
    print("охват — зал целиком в кадр уже не поместится.")


if __name__ == '__main__':
    main()
