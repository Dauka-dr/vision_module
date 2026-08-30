"""Runs the trained ST-GCN++ on the pose buffers the pipeline accumulates.

Preprocessing is taken from the model's own config rather than reimplemented, so
a live buffer goes through exactly the same steps as the clips it was trained on
— normalisation, frame sampling, feature generation. Rebuilding that by hand is
the usual way a model that scored well offline quietly misbehaves live.
"""
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Order must match the training labels in build_subset.py.
TARGET7 = ['phone', 'writing', 'raising_hand', 'talking', 'walking',
           'running', 'fighting']
TARGET7_RU = {
    'phone': 'смотрит в телефон',
    'writing': 'пишет / ноутбук',
    'raising_hand': 'поднимает руку',
    'talking': 'разговаривает',
    'walking': 'идёт',
    'running': 'бежит',
    'fighting': 'дерётся',
}


class ActionClassifier:
    """Classifies a (T, 17, 3) pose sequence into one of the trained actions."""

    def __init__(self,
                 config: str = 'configs/stgcnpp_target7.py',
                 checkpoint: str = 'models/stgcnpp_target7.pth',
                 classes: Optional[List[str]] = None,
                 device: str = 'cuda:0'):
        from mmaction.apis import init_recognizer

        config = os.path.join(PROJECT_DIR, config) if not os.path.isabs(config) else config
        checkpoint = (os.path.join(PROJECT_DIR, checkpoint)
                      if not os.path.isabs(checkpoint) else checkpoint)

        init_default_scope('mmaction')
        self.model = init_recognizer(config, checkpoint, device=device)
        self.classes = classes or TARGET7
        self.device = device

        cfg = Config.fromfile(config)
        # The val pipeline, minus the steps that read an annotation file.
        steps = [s for s in cfg.val_pipeline
                 if s['type'] not in ('DecompressPose',)]
        self.pipeline = Compose(steps)

    def _to_sample(self, sequence: np.ndarray, img_shape: Tuple[int, int]):
        """Wrap a buffer as the annotation dict the pipeline expects.

        Accepts (T, 17, 3) for one person or (M, T, 17, 3) for a group. Three of
        the seven classes — fighting, talking, walking — are two-person actions in
        NTU and were trained with both skeletons present, so feeding a lone person
        for those throws away the interaction the model relies on.
        """
        if sequence.ndim == 3:
            sequence = sequence[None]
        kpt = sequence[..., :2].astype(np.float32)      # (M, T, 17, 2)
        score = sequence[..., 2].astype(np.float32)     # (M, T, 17)
        return dict(
            keypoint=kpt,
            keypoint_score=score,
            total_frames=sequence.shape[1],
            frame_dir='live',
            img_shape=img_shape,
            original_shape=img_shape,
            start_index=0,
            modality='Pose',
            label=-1,
        )

    def __call__(self, sequences: List[np.ndarray],
                 img_shape: Tuple[int, int]) -> List[Tuple[str, float]]:
        """Classify several people at once; returns (class, confidence) each."""
        if not sequences:
            return []
        init_default_scope('mmaction')
        batch = pseudo_collate([self.pipeline(self._to_sample(s, img_shape))
                                for s in sequences])
        with torch.no_grad():
            results = self.model.test_step(batch)

        out = []
        for res in results:
            # pred_score is already normalised by the recognizer's head; running
            # softmax again would flatten every confidence towards uniform.
            scores = res.pred_score.cpu().numpy()
            idx = int(scores.argmax())
            out.append((self.classes[idx], float(scores[idx])))
        return out
