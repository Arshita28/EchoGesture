from collections import deque
import numpy as np


class Stabilizer:
    """Require consecutive confident frames; invalid frames clear stale state."""
    def __init__(self, frames=4):
        self.history = deque(maxlen=frames)
        self.frames = frames

    def update(self, label):
        if label is None:
            self.history.clear()
            return None
        self.history.append(label)
        return label if len(self.history) == self.frames and len(set(self.history)) == 1 else None


def classify(bundle, features):
    model = bundle['model']
    scores = model.predict_proba(np.asarray(features).reshape(1, -1))[0]
    index = int(np.argmax(scores))
    confidence = float(scores[index])
    accepted = confidence >= bundle['threshold']
    return {'status': 'ok' if accepted else 'uncertain',
            'label': str(model.classes_[index]) if accepted else None,
            'candidate': str(model.classes_[index]),
            'threshold': bundle['threshold'], 'confidence': confidence}
