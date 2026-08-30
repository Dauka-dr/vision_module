"""Per-class results and the confusion matrix for the 7-class model.

An aggregate score hides the thing that actually matters here: whether "phone"
and "writing" get mixed up. The skeleton shows the same posture for both — head
down, hands in front — so this is the pair that decides whether the object stream
is needed. The matrix answers it with counts instead of argument.
"""
# Run from anywhere: the project root holds the configs and weights.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)
import argparse
import os

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner

CLASSES = ['phone', 'writing', 'raising_hand', 'talking', 'walking',
           'running', 'fighting']
RU = {'phone': 'телефон', 'writing': 'пишет', 'raising_hand': 'рука',
      'talking': 'говорит', 'walking': 'идёт', 'running': 'бежит',
      'fighting': 'дерётся'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/stgcnpp_target7.py')
    ap.add_argument('--checkpoint', default='models/stgcnpp_target7.pth')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    cfg.work_dir = 'work_dirs/eval_tmp'
    cfg.test_dataloader.batch_size = 16
    # The runner would otherwise log a full training banner for a pure eval.
    cfg.log_level = 'ERROR'

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)
    model = runner.model.eval()

    preds, gts = [], []
    with torch.no_grad():
        for batch in runner.test_dataloader:
            data = model.data_preprocessor(batch, training=False)
            out = model(**data, mode='predict')
            for sample in out:
                preds.append(int(sample.pred_score.argmax()))
                gts.append(int(sample.gt_label))

    preds, gts = np.array(preds), np.array(gts)
    n = len(CLASSES)
    cm = np.zeros((n, n), dtype=int)
    for g, p in zip(gts, preds):
        cm[g, p] += 1

    print(f"клипов в валидации: {len(gts)}")
    print(f"общая точность:     {(preds == gts).mean():.2%}\n")

    print("точность по классам:")
    print(f"  {'класс':<12}{'верно':>7}{'всего':>7}{'recall':>9}{'precision':>11}")
    print("  " + "-" * 46)
    for i, c in enumerate(CLASSES):
        recall = cm[i, i] / max(cm[i].sum(), 1)
        precision = cm[i, i] / max(cm[:, i].sum(), 1)
        print(f"  {RU[c]:<12}{cm[i, i]:>7}{cm[i].sum():>7}"
              f"{recall:>9.1%}{precision:>11.1%}")

    print("\nматрица ошибок (строка = истина, столбец = предсказание):")
    header = "  " + " " * 12 + "".join(f"{RU[c][:7]:>9}" for c in CLASSES)
    print(header)
    for i, c in enumerate(CLASSES):
        row = "".join(f"{cm[i, j]:>9}" if i != j else f"{cm[i, j]:>8}*"
                      for j in range(n))
        print(f"  {RU[c]:<12}{row}")
    print("  (* — диагональ, верные ответы)")

    ph, wr = CLASSES.index('phone'), CLASSES.index('writing')
    mixed = cm[ph, wr] + cm[wr, ph]
    total = cm[ph].sum() + cm[wr].sum()
    print(f"\nключевая пара «телефон <-> пишет»:")
    print(f"  телефон принят за 'пишет':  {cm[ph, wr]} из {cm[ph].sum()}")
    print(f"  'пишет' принят за телефон:  {cm[wr, ph]} из {cm[wr].sum()}")
    print(f"  доля путаницы в этой паре:  {mixed / max(total, 1):.2%}")

    worst = sorted(((cm[i, j], i, j) for i in range(n) for j in range(n)
                    if i != j), reverse=True)[:5]
    print("\nсамые частые ошибки:")
    for cnt, i, j in worst:
        if cnt:
            print(f"  {RU[CLASSES[i]]} -> {RU[CLASSES[j]]}: {cnt}")


if __name__ == '__main__':
    main()
