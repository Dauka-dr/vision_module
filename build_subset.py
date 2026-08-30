"""Build a training set for the target classes out of NTU120.

NTU is labelled with its own 120 actions, so the target taxonomy has to be mapped
onto it. Several target classes map to more than one NTU action (fighting covers
punching, kicking and pushing), and four of them — the static states — have no NTU
equivalent at all and are handled by geometry instead, not here.

    python build_subset.py            # writes dataset/annotations/target_7cls.pkl
    python build_subset.py --list     # just show what maps to what
"""
import argparse
import os
import pickle
from collections import Counter

# NTU action ids are 1-based in the papers (A029); labels in the pkl are 0-based.
# Kept in the A-form here because that is how the dataset documents them, and
# converted once below — mixing the two conventions is an easy way to build a
# silently wrong dataset.
TARGET_CLASSES = [
    ('phone',        [28, 29],       'смотрит в телефон'),
    ('writing',      [11, 12, 30],   'пишет / работает с ноутбуком'),
    ('raising_hand', [23],           'поднимает руку'),
    ('talking',      [117],          'разговаривает с соседом'),
    ('walking',      [59, 60],       'идёт'),
    ('running',      [99],           'бежит'),
    ('fighting',     [50, 51, 52],   'дерётся'),
]

# The four target classes with no NTU counterpart. NTU is built out of
# transitions ("sit down", "stand up"), never sustained states, so these cannot
# be learned from it at any sample count.
GEOMETRIC_CLASSES = [
    ('sitting_attentive', 'сидит, внимателен'),
    ('sleeping',          'спит / голова на парте'),
    ('sitting',           'сидит'),
    ('standing',          'стоит'),
]


def load_labels(path='mmaction2/tools/data/skeleton/label_map_ntu60.txt'):
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return {i + 1: line.strip() for i, line in enumerate(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/skeleton/ntu120_2d.pkl')
    ap.add_argument('--out', default='dataset/annotations/target_7cls.pkl')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    names = load_labels()
    if args.list:
        print("обучаемые классы (из NTU120):")
        for key, actions, ru in TARGET_CLASSES:
            src = ', '.join(names.get(a, f'A{a:03d}') for a in actions)
            print(f"  {ru:<30} <- {src}")
        print("\nгеометрические классы (в NTU отсутствуют):")
        for key, ru in GEOMETRIC_CLASSES:
            print(f"  {ru}")
        return

    print(f"читаю {args.src} ...", flush=True)
    with open(args.src, 'rb') as f:
        data = pickle.load(f)

    # NTU action id -> our 0-based class index
    action_to_class, class_names = {}, []
    for idx, (key, actions, ru) in enumerate(TARGET_CLASSES):
        class_names.append(key)
        for a in actions:
            action_to_class[a - 1] = idx        # pkl labels are 0-based

    kept = []
    for ann in data['annotations']:
        cls = action_to_class.get(ann['label'])
        if cls is None:
            continue
        ann = dict(ann)
        ann['label'] = cls
        kept.append(ann)

    # Split by performer, not by clip. NTU's own xsub split was drawn for all 120
    # actions at once, and the actions added in NTU120 were filmed by a different
    # set of performers — using it here left two classes with more validation
    # clips than training ones. Splitting the performers of *this* subset keeps
    # the ratio even, and still guarantees nobody appears on both sides, which is
    # what stops the score being inflated by the model recognising people.
    def performer(frame_dir):
        i = frame_dir.find('P')
        return frame_dir[i:i + 4]

    # Hold out a quarter of the performers. Picked with a fixed seed rather than
    # by stride so the split does not line up with NTU's own performer numbering,
    # which correlates with recording setup.
    import random
    performers = sorted({performer(a['frame_dir']) for a in kept})
    rng = random.Random(0)
    shuffled = performers[:]
    rng.shuffle(shuffled)
    val_performers = set(shuffled[:max(1, round(len(performers) * 0.25))])

    new_train, new_val = [], []
    for ann in kept:
        (new_val if performer(ann['frame_dir']) in val_performers
         else new_train).append(ann['frame_dir'])

    print(f"исполнителей: {len(performers)}, из них в val: {len(val_performers)}")

    out = dict(split=dict(xsub_train=new_train, xsub_val=new_val),
               annotations=kept)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(out, f)

    print(f"\n{'класс':<16}{'всего':>8}{'train':>8}{'val':>7}")
    print('-' * 40)
    tr_set, va_set = set(new_train), set(new_val)
    for idx, key in enumerate(class_names):
        clips = [a for a in kept if a['label'] == idx]
        t = sum(1 for a in clips if a['frame_dir'] in tr_set)
        v = sum(1 for a in clips if a['frame_dir'] in va_set)
        print(f"{key:<16}{len(clips):>8}{t:>8}{v:>7}")
    print('-' * 40)
    print(f"{'ИТОГО':<16}{len(kept):>8}{len(new_train):>8}{len(new_val):>7}")
    print(f"\nзаписано: {args.out}")
    print(f"классов: {len(class_names)} -> {class_names}")


if __name__ == '__main__':
    main()
