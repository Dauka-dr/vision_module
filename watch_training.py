"""Live view of a training run, read from its own log — safe to run alongside.

Reads the scalars mmengine writes as it trains, so it neither touches the run nor
slows it down. Ctrl+C stops watching, not training.

    python watch_training.py                 # latest run
    python watch_training.py --once          # print once and exit
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import timedelta

# The Windows console defaults to a legacy codepage that mangles or refuses
# Cyrillic; switch it and the stream to UTF-8, tolerating consoles that cannot.
if os.name == 'nt':
    os.system('chcp 65001 >nul 2>&1')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass

# ASCII ramp rather than block characters: the Windows console runs cp1251 here
# and cannot encode them.
BAR = "._-=+*#@"


def latest_run(work_dir):
    runs = sorted(glob.glob(os.path.join(work_dir, '*', 'vis_data', 'scalars.json')))
    if not runs:
        raise SystemExit(f"нет логов обучения в {work_dir}")
    return runs[-1]


def read(path):
    train, val = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written last line while training appends
            (val if 'acc/top1' in rec else train).append(rec)
    return train, val


def spark(values, width=40):
    """Compact trend line, so the shape of the curve is visible in a terminal."""
    if not values:
        return ''
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return ''.join(BAR[min(int((v - lo) / span * (len(BAR) - 1)), len(BAR) - 1)]
                   for v in vals)


def render(path, iters_per_epoch, total_epochs, reference=None):
    train, val = read(path)
    if not train:
        return "лог пуст — обучение ещё не начало писать"

    cur = train[-1]
    # mmengine's "iter" here counts from the start of the run, not from the start
    # of the epoch, so it is already the global step.
    epoch, done = cur.get('epoch', 0), cur.get('iter', 0)
    in_epoch = done - (epoch - 1) * iters_per_epoch
    total = total_epochs * iters_per_epoch
    pct = done / total * 100 if total else 0

    sec_per_iter = cur.get('time', 0)
    eta = timedelta(seconds=int((total - done) * sec_per_iter))

    losses = [r['loss'] for r in train if 'loss' in r]
    accs = [r['top1_acc'] for r in train if 'top1_acc' in r]

    out = []
    out.append(f"эпоха {epoch}/{total_epochs}   итерация {in_epoch}/{iters_per_epoch}"
               f"   всего {pct:.1f}%")
    filled = int(pct / 100 * 44)
    out.append("[" + "#" * filled + "." * (44 - filled) + "]")
    out.append("")
    out.append(f"loss        {losses[-1]:6.3f}   {spark(losses)}")
    if accs:
        out.append(f"train top1  {accs[-1]:6.1%}   {spark(accs)}")
    out.append("")
    out.append(f"скорость    {sec_per_iter:.3f} с/итерация"
               f"   память {cur.get('memory', 0)} МБ")
    out.append(f"осталось    ~{eta}")

    if val:
        out.append("")
        out.append("валидация по эпохам:")
        for rec in val[-8:]:
            top1 = rec.get('acc/top1', 0)
            top5 = rec.get('acc/top5', 0)
            out.append(f"   эпоха {rec.get('step', '?'):>3}   "
                       f"top1 {top1:.2%}   top5 {top5:.2%}")
        best = max(r.get('acc/top1', 0) for r in val)
        line = f"   лучшая top1: {best:.2%}"
        if reference:
            line += f"   (эталон: {reference:.2%})"
        out.append(line)
    else:
        out.append("")
        out.append("валидация будет после первой эпохи")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-dir', default='work_dirs/stgcnpp_ntu60_1gpu')
    ap.add_argument('--iters-per-epoch', type=int, default=12529)
    ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--interval', type=float, default=10)
    ap.add_argument('--reference', type=float, default=None,
                    help="эталонная top1 для сравнения, напр. 0.8929")
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()

    path = latest_run(args.work_dir)
    print(f"лог: {path}\n")

    if args.once:
        print(render(path, args.iters_per_epoch, args.epochs, args.reference))
        return

    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"{os.path.basename(os.path.dirname(os.path.dirname(path)))}"
                  f"   обновление каждые {args.interval:.0f}с   Ctrl+C — выйти\n")
            print(render(path, args.iters_per_epoch, args.epochs, args.reference))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nнаблюдение остановлено (обучение продолжается)")


if __name__ == '__main__':
    main()
