"""Final demo: camera, video or image -> per-person action and posture.

    python demo.py                        # webcam
    python demo.py --source video.mp4 --save out.mp4
    python demo.py --source photo.jpg
    python demo.py --no-action            # posture only, skips loading the model
"""
import argparse
import time

import cv2
import numpy as np

from pipeline import COCO_SKELETON
from recognizer import Recognizer

PALETTE = [(255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
           (207, 210, 49), (72, 249, 10), (26, 147, 52), (0, 212, 187),
           (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255),
           (0, 24, 236), (132, 56, 255), (82, 0, 133), (203, 56, 255)]

# Latin labels: OpenCV's built-in font cannot draw Cyrillic, and rendering a
# TTF per frame would cost more than the pipeline itself.
LABEL_EN = {
    'phone': 'PHONE', 'writing': 'WRITING', 'raising_hand': 'HAND UP',
    'talking': 'TALKING', 'walking': 'WALKING', 'running': 'RUNNING',
    'fighting': 'FIGHTING', 'sitting': 'sitting', 'standing': 'standing',
    'sitting_attentive': 'attentive', 'sleeping': 'SLEEPING',
    'unknown': '?',
}
# Actions are highlighted; postures are the quiet default.
ALERT = {'phone', 'fighting', 'sleeping', 'running'}


def draw(frame, people, kpt_thr=0.3):
    vis = frame.copy()
    for p in people:
        color = PALETTE[p.track_id % len(PALETTE)]
        x1, y1, x2, y2 = p.box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        for i, j in COCO_SKELETON:
            if p.kpt_scores[i] > kpt_thr and p.kpt_scores[j] > kpt_thr:
                cv2.line(vis, tuple(p.keypoints[i].astype(int)),
                         tuple(p.keypoints[j].astype(int)), color, 2)
        for kpt, s in zip(p.keypoints, p.kpt_scores):
            if s > kpt_thr:
                cv2.circle(vis, tuple(kpt.astype(int)), 3, color, -1)

        text = LABEL_EN.get(p.label, p.label)
        conf = p.action_conf if p.action else p.state_conf
        text = f"#{p.track_id} {text} {conf:.2f}"
        text_color = (0, 0, 255) if p.label in ALERT else (255, 255, 255)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis, (x1, max(y1 - th - 8, 0)), (x1 + tw + 4, max(y1, th)),
                      (0, 0, 0), -1)
        cv2.putText(vis, text, (x1 + 2, max(y1 - 5, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

        if p.objects:
            names = ",".join(o[0] for o in p.objects[:2])
            cv2.putText(vis, names, (x1, min(y2 + 16, vis.shape[0] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='0')
    ap.add_argument('--scene', default='auto',
                    choices=['auto', 'webcam', 'auditorium'])
    ap.add_argument('--seq-len', type=int, default=100,
                    help="frames of pose history before an action is predicted")
    ap.add_argument('--action-thr', type=float, default=0.6)
    ap.add_argument('--no-action', action='store_true',
                    help="posture only — skips loading the action model")
    ap.add_argument('--save', default=None)
    args = ap.parse_args()

    t0 = time.time()
    rec = Recognizer(scene=args.scene, seq_len=args.seq_len,
                     action_thr=args.action_thr, with_action=not args.no_action)
    print(f"готово за {time.time() - t0:.1f}s", flush=True)

    if args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        frame = cv2.imread(args.source)
        people = rec.process(frame)
        print(f"{len(people)} человек; действия появятся только на видео "
              f"(нужно {args.seq_len} кадров истории)")
        for p in people[:10]:
            print(f"  #{p.track_id:<3} {p.label_ru}")
        out = args.save or 'demo_out.jpg'
        cv2.imwrite(out, draw(frame, people))
        print(f"записано: {out}")
        return

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source,
                           cv2.CAP_DSHOW if isinstance(source, int) else cv2.CAP_ANY)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"не удалось открыть источник {args.source}")
        return

    writer, n, fps_ema = None, 0, None
    print("идёт обработка, 'q' в окне — выход", flush=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time()
        people = rec.process(frame)
        dt = time.time() - t
        fps_ema = 1 / dt if fps_ema is None else 0.9 * fps_ema + 0.1 / dt

        vis = draw(frame, people)
        acting = sum(bool(p.action) for p in people)
        cv2.putText(vis, f"FPS {fps_ema:.1f} | people {len(people)} | "
                         f"actions {acting}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if args.save:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(args.save,
                                         cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
            writer.write(vis)

        cv2.imshow('action recognition (q to quit)', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        n += 1
        if n % 30 == 0:
            labels = {}
            for p in people:
                labels[p.label] = labels.get(p.label, 0) + 1
            print(f"[{n}] fps {fps_ema:.1f}  {labels}", flush=True)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"обработано кадров: {n}")


if __name__ == '__main__':
    main()
