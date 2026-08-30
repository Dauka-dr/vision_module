"""The finished system: pose pipeline + action model + posture rules -> 11 classes.

The eleven target classes come from two different places, and the split is not
arbitrary. Seven are *actions* — they have a temporal signature the network was
trained on. Four are *sustained states*, which NTU does not contain at all and
which are derived from skeleton geometry instead.

Both are reported, because they answer different questions: a person is always in
some posture, and may additionally be doing something. `label` picks whichever is
the more informative answer for display.

    from recognizer import Recognizer
    rec = Recognizer()
    for frame in video:
        for person in rec.process(frame):
            print(person.track_id, person.label, person.action, person.state)
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from action_classifier import ActionClassifier, TARGET7_RU
from pipeline import PosePipeline
from states import StateClassifier, STATES_RU

# Actions the model only recognises with a second person present: they are
# interactions, and NTU trained them with both skeletons in frame.
PAIRED_ACTIONS = {'fighting', 'talking', 'walking'}

# Objects that argue for one of the two postures the skeleton cannot separate.
# "Phone" and "writing" put the body in the same shape — head down, hands
# forward — and on held-out NTU clips they accounted for 87% of all the model's
# errors. What differs is the thing in the person's hands.
PHONE_EVIDENCE = {'cell_phone'}
WRITING_EVIDENCE = {'book', 'laptop', 'keyboard'}


def arbitrate_with_objects(action: str, conf: float, objects) -> Tuple[str, float]:
    """Let a detected object overrule a phone/writing call the pose cannot settle.

    The evidence is deliberately one-directional. A detected phone is a strong
    argument; a *missing* phone is not, because the object stream only reaches
    about a third of people — an absent detection usually means it was not seen,
    not that it is not there. So this only ever fires on positive evidence.
    """
    if action not in ('phone', 'writing') or not objects:
        return action, conf
    names = {name for name, _, _ in objects}
    if action == 'phone' and not (names & PHONE_EVIDENCE) and (names & WRITING_EVIDENCE):
        return 'writing', min(conf, 0.75)
    if action == 'writing' and (names & PHONE_EVIDENCE):
        return 'phone', min(conf, 0.75)
    return action, conf


@dataclass
class Recognized:
    track_id: int
    box: np.ndarray
    keypoints: np.ndarray
    kpt_scores: np.ndarray
    objects: List[tuple]
    state: str                      # always available once a person is tracked
    state_conf: float
    action: Optional[str] = None    # only once the pose buffer is full
    action_conf: float = 0.0
    label: str = ''                 # the one answer to show
    # Stays None unless an identity provider is attached *and* the face is large
    # enough to recognise — on a wide hall shot that is a small minority.
    identity: Optional[str] = None
    identity_conf: float = 0.0

    @property
    def who(self) -> str:
        return self.identity or f"#{self.track_id}"

    @property
    def label_ru(self) -> str:
        return TARGET7_RU.get(self.label) or STATES_RU.get(self.label, self.label)


class Recognizer:
    """Full stack: detect, pose, track, classify action, derive posture."""

    def __init__(self, scene: str = 'auto', seq_len: int = 100,
                 action_thr: float = 0.6, pair_dist: float = 3.0,
                 with_action: bool = True, identity_provider=None,
                 device: str = 'cuda:0'):
        self.pipe = PosePipeline(scene=scene, seq_len=seq_len, device=device)
        self.states = StateClassifier()
        self.action = ActionClassifier(device=device) if with_action else None
        # Optional: any object exposing identify(frame, box, keypoints, scores).
        # See identity.py for adapters and for why most people stay anonymous.
        self.identity = None
        if identity_provider is not None:
            from identity import IdentityResolver
            self.identity = IdentityResolver(identity_provider)
        self.action_thr = action_thr
        # How far a neighbour may be, in person-heights, to count as taking part
        # in the same interaction.
        self.pair_dist = pair_dist
        self._last_action = {}

    @property
    def tracker(self):
        return self.pipe.tracker

    def _neighbour(self, i: int, people) -> Optional[int]:
        """Nearest other person, if close enough to plausibly interact."""
        if len(people) < 2:
            return None
        centre = lambda p: np.array([(p.box[0] + p.box[2]) / 2,
                                     (p.box[1] + p.box[3]) / 2])
        here = centre(people[i])
        height = max(people[i].box[3] - people[i].box[1], 1.0)
        best, best_d = None, np.inf
        for j, other in enumerate(people):
            if j == i:
                continue
            d = np.linalg.norm(centre(other) - here) / height
            if d < best_d:
                best, best_d = j, d
        return best if best_d <= self.pair_dist else None

    def process(self, frame: np.ndarray) -> List[Recognized]:
        people = self.pipe.process(frame)
        out = []

        # --- posture, available from the first frame ---
        for p in people:
            state, conf = self.states(p.track_id, p.box, p.keypoints, p.kpt_scores)
            out.append(Recognized(
                track_id=p.track_id, box=p.box, keypoints=p.keypoints,
                kpt_scores=p.kpt_scores, objects=p.objects,
                state=state, state_conf=conf))

        # --- action, once a person has a full pose buffer ---
        if self.action is not None:
            ready = [i for i, p in enumerate(people) if p.sequence is not None]
            if ready:
                seqs = []
                for i in ready:
                    seq = people[i].sequence
                    j = self._neighbour(i, people)
                    # Pair the person with a neighbour so interaction classes see
                    # what they were trained on; alone they score far worse
                    # (85% vs 99% measured on held-out NTU clips).
                    if j is not None and people[j].sequence is not None:
                        seqs.append(np.stack([seq, people[j].sequence]))
                    else:
                        seqs.append(seq)
                preds = self.action(seqs, img_shape=frame.shape[:2])
                for i, (cls, conf) in zip(ready, preds):
                    # A paired action claimed for a person standing alone is not
                    # believable — the model is extrapolating from one skeleton.
                    if cls in PAIRED_ACTIONS and self._neighbour(i, people) is None:
                        continue
                    if conf >= self.action_thr:
                        cls, conf = arbitrate_with_objects(cls, conf,
                                                           people[i].objects)
                        out[i].action, out[i].action_conf = cls, conf
                        self._last_action[people[i].track_id] = (cls, conf)

        # --- identity, at most a few attempts per track ---
        if self.identity is not None:
            self.identity.resolve(frame, out)
            for r in out:
                rec = self.identity.get(r.track_id)
                r.identity, r.identity_conf = rec.name, rec.confidence

        # --- pick the single label to display ---
        for r in out:
            head_down = r.state == 'sleeping'
            state = self.states.refine_attentive(r.state, r.objects, head_down)
            r.state = state
            # An action is the more specific answer, so it wins when confident.
            r.label = r.action if r.action else state
        return out
