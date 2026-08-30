"""Fetches everything the repository cannot carry itself.

Model weights and the test scene are too large to keep in git, so a fresh clone
has holes in it. This fills them, and is safe to re-run — anything already
present is left alone.

    python fetch_assets.py            # всё, что нужно для запуска
    python fetch_assets.py --check    # только показать, чего не хватает
"""
import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# name -> (relative path, url, size hint, what it is for)
ASSETS = {
    'detector': (
        'models/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth',
        'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/'
        'rtmdet_s_8xb32-300e_coco/'
        'rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth',
        '87 МБ', 'детекция людей и объектов'),
    'pose': (
        'models/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/'
        'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
        '52 МБ', 'скелет, 17 точек'),
    'scene': (
        'test_scenes/auditorium_bi.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/3/3c/'
        'BI_Norwegian_Business_School_Auditorium_7DM29184.jpg',
        '13 МБ', 'эталонная сцена для метрик'),
}

# Present in the repository itself — checked, never downloaded.
IN_REPO = {
    'models/stgcnpp_target7.pth':
        'классификатор действий, 7 классов (обучен нами)',
    'models/stgcnpp_ntu60_epoch12.pth':
        'backbone для дообучения на своих классах (обучен нами)',
}

# Alternative model sizes. The pipeline fetches these itself the moment someone
# selects one (`--detector rtmdet-t`), so they are not needed to run. The metrics
# scripts, however, address the files directly and fail if they are absent, so
# --all exists to reproduce the full benchmark set.
ALTERNATIVES = {
    'rtmdet-t': (
        'models/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
        'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/'
        'rtmdet_tiny_8xb32-300e_coco/'
        'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
        '55 МБ', 'самый быстрый детектор'),
    'rtmdet-m': (
        'models/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth',
        'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/'
        'rtmdet_m_8xb32-300e_coco/'
        'rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth',
        '214 МБ', 'детектор побольше'),
    'rtmpose-t': (
        'models/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.pth',
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/'
        'rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.pth',
        '13 МБ', 'самая быстрая поза'),
    'rtmpose-s': (
        'models/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth',
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/'
        'rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth',
        '22 МБ', 'поза поменьше'),
}

# Needed only to retrain, and rebuilt locally rather than downloaded.
OPTIONAL = {
    'dataset/annotations/target_7cls.pkl':
        'выборка NTU для переобучения — собрать: python build_subset.py',
}


def download(url: str, dest: str):
    import torch
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    torch.hub.download_url_to_file(url, dest, progress=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='только проверить, ничего не качать')
    ap.add_argument('--all', action='store_true',
                    help='вместе с запасными размерами моделей (для метрик)')
    args = ap.parse_args()

    os.chdir(PROJECT_DIR)
    missing = []

    print("обязательное для запуска:")
    for path, what in IN_REPO.items():
        ok = os.path.exists(path)
        print(f"  [{'x' if ok else ' '}] {path}")
        print(f"      {what}")
        if not ok:
            print("      ОТСУТСТВУЕТ — этот файл лежит в репозитории, "
                  "проверьте полноту клона")

    for name, (path, url, size, what) in ASSETS.items():
        ok = os.path.exists(path)
        print(f"  [{'x' if ok else ' '}] {path}  ({size})")
        print(f"      {what}")
        if not ok:
            missing.append((name, path, url, size))

    hint = ('скачиваются' if args.all
            else 'по --all; иначе подтянутся сами при выборе')
    print(f"\nзапасные размеры моделей ({hint}):")
    for name, (path, url, size, what) in ALTERNATIVES.items():
        ok = os.path.exists(path)
        print(f"  [{'x' if ok else ' '}] {name:<11}{size:>8}   {what}")
        if not ok and args.all:
            missing.append((name, path, url, size))

    print("\nтолько для переобучения:")
    for path, what in OPTIONAL.items():
        ok = os.path.exists(path)
        print(f"  [{'x' if ok else ' '}] {path}")
        print(f"      {what}")

    if not missing:
        print("\nвсё на месте.")
        return

    if args.check:
        total = ', '.join(f'{n} ({s})' for n, _, _, s in missing)
        print(f"\nне хватает: {total}")
        print("скачать: python fetch_assets.py"
              + ("" if args.all else "   (или --all вместе с запасными)"))
        return

    print(f"\nскачиваю {len(missing)} файл(ов)...")
    for name, path, url, size in missing:
        print(f"\n{name} -> {path} ({size})")
        try:
            download(url, path)
        except Exception as exc:
            print(f"  не удалось: {exc}")
            print(f"  ссылка: {url}")
            sys.exit(1)
    print("\nготово. запуск: python demo.py")


if __name__ == '__main__':
    main()
