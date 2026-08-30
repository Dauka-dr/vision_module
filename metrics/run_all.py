"""Runs every metric in this folder and saves the output under results/.

Some metrics need weights or datasets that were deleted after they had served
their purpose — the RTMDet-l/x comparison, the RTMO comparison, the NTU source
pickles. Those are reported as skipped with the reason rather than failing the
run, so a full pass still works on a fresh checkout.

    python metrics/run_all.py              # everything available
    python metrics/run_all.py --list       # what would run, and what is missing
    python metrics/run_all.py --only stability autocal
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')

# name -> (script, args, what it needs, what it answers)
METRICS = {
    'detection': (
        'bench_detection.py', [],
        ['models/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'],
        'разрешение против тайлинга'),
    'modelsize': (
        'bench_modelsize.py', [],
        ['models/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'],
        'размер модели против разрешения'),
    'fair': (
        'bench_fair.py', [],
        ['models/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'],
        'честное сравнение детекторов с единым NMS'),
    'threshold': (
        'bench_threshold.py', [],
        ['models/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'],
        'развёртка по порогу уверенности'),
    'pose': (
        'bench_pose.py', [],
        ['models/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth'],
        'двухстадийная схема против RTMO'),
    'pipeline': (
        'bench_pipeline.py', [],
        ['models/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'],
        'пропускная способность по конфигурациям'),
    'objects': (
        'bench_objects.py', [],
        ['models/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'],
        'детекция телефонов, книг, ноутбуков'),
    'object_crop': (
        'bench_object_crop.py', [],
        ['models/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'],
        'кроп-и-увеличение для объектов (тупик)'),
    'stability': (
        'bench_stability.py', [],
        ['models/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'],
        'нестабильность выхода между кадрами'),
    'autocal': (
        'bench_autocal.py', [],
        ['models/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'],
        'автокалибровка под разные разрешения'),
    'crowd100': (
        'bench_crowd100.py', [],
        ['models/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'],
        'поведение системы на ~100 людях'),
    'tracking': (
        'test_tracking.py', ['static'],
        ['models/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'],
        'стабильность трекинга'),
    'confusion': (
        'eval_confusion.py', [],
        ['models/stgcnpp_target7.pth', 'dataset/annotations/target_7cls.pkl'],
        'матрица ошибок классификатора действий'),
    'face_budget': (
        'face_budget.py', [],
        [],   # чистая геометрия, ни весов, ни данных не нужно
        'дальность распознавания лица по разрешению камеры'),
}

COMMON = ['test_scenes/auditorium_bi.jpg']


def missing(paths):
    return [p for p in paths if not os.path.exists(os.path.join(ROOT, p))]


# mmengine narrates every checkpoint load and deprecation; none of it is a result.
NOISE = ('Loads checkpoint', 'state dict do not match', 'unexpected key',
         'mmengine - WARNING', 'UserWarning', 'warnings.warn', 'meshgrid',
         '_VF.meshgrid', 'does not exist', 'will be used instead',
         'NormalInit', 'TruncNormalInit', 'initialize', '- mmengine - INFO')


def strip_noise(text: str) -> str:
    out, after_noise, blank = [], False, False
    for ln in text.splitlines():
        if any(n in ln for n in NOISE):
            # A stripped line usually leaves its trailing blank behind, which
            # would otherwise split tables apart.
            after_noise = True
            continue
        if not ln.strip():
            if after_noise or blank:
                continue
            out.append('')
            blank = True
        else:
            out.append(ln)
            blank = False
        after_noise = False
    return '\n'.join(out).strip() + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='+', help='запустить только эти метрики')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    names = args.only or list(METRICS)
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise SystemExit(f"неизвестные метрики: {unknown}\nдоступны: {list(METRICS)}")

    if args.list:
        print(f"{'метрика':<14}{'статус':<12}что измеряет")
        print('-' * 74)
        for name in names:
            script, _, needs, what = METRICS[name]
            lack = missing(needs + COMMON)
            status = 'готово' if not lack else 'нет данных'
            print(f"{name:<14}{status:<12}{what}")
            if lack:
                for p in lack:
                    print(f"{'':<26}нужен {p}")
        return

    os.makedirs(RESULTS, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = []

    for name in names:
        script, extra, needs, what = METRICS[name]
        lack = missing(needs + COMMON)
        if lack:
            print(f"[ пропуск ] {name:<12} {what}")
            print(f"{'':<13}нет: {', '.join(lack)}")
            summary.append((name, 'пропущено', ', '.join(lack)))
            continue

        print(f"[ запуск  ] {name:<12} {what}", flush=True)
        t0 = time.time()
        # Without this the child writes Russian output in the console codepage
        # and the captured text comes back as mojibake.
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, script), *extra],
            cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
            errors='replace', env=env)
        dt = time.time() - t0

        out_path = os.path.join(RESULTS, f'{name}.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"# {what}\n# {script} {' '.join(extra)}\n"
                    f"# запуск {stamp}, {dt:.0f}s\n\n")
            f.write(strip_noise(proc.stdout))
            if proc.returncode != 0:
                f.write(f"\n--- stderr ---\n{proc.stderr[-4000:]}")

        ok = proc.returncode == 0
        print(f"{'':<13}{'готово' if ok else 'ОШИБКА'} за {dt:.0f}s "
              f"-> results/{name}.txt", flush=True)
        summary.append((name, 'готово' if ok else 'ошибка', f'{dt:.0f}s'))

    print(f"\n{'метрика':<14}{'итог':<12}")
    print('-' * 50)
    for name, status, note in summary:
        print(f"{name:<14}{status:<12}{note}")
    print(f"\nрезультаты: metrics/results/")


if __name__ == '__main__':
    main()
