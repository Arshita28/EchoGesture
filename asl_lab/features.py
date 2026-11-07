import numpy as np

LETTERS = tuple(c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c not in 'JZ')
FEATURE_VERSION = 'wrist-max-radius-aspect-v2'


def normalize(points, image_aspect=1.0):
    """Translation/scale normalization; deliberately retain hand orientation."""
    p = np.array(points, dtype=np.float64, copy=True)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        raise ValueError('Expected 21 finite xyz landmarks')
    if not np.isfinite(image_aspect) or image_aspect <= 0:
        raise ValueError('Expected positive image width/height aspect ratio')
    # MediaPipe x/z use image-width units; y uses image-height units.
    # Convert y to width units before geometry normalization.
    p[:, 1] /= image_aspect
    p = p - p[0]
    scale = np.linalg.norm(p, axis=1).max()
    if scale < 1e-8:
        raise ValueError('Degenerate hand landmarks')
    return (p / scale).reshape(63)


class HandExtractor:
    def __init__(self, static_image_mode=True):
        import mediapipe as mp
        # Pinned MediaPipe release includes these pretrained models locally.
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.points = None
        self.handedness = None

    def extract(self, rgb):
        self.points = None
        self.handedness = None
        result = self.hands.process(np.ascontiguousarray(rgb))
        hands = result.multi_hand_landmarks or []
        if len(hands) != 1:
            return None, 'no_hand' if not hands else 'multiple_hands'
        points = [[p.x, p.y, p.z] for p in hands[0].landmark]
        self.points = points
        classification = result.multi_handedness[0].classification[0]
        self.handedness = {'label': classification.label, 'score': float(classification.score)}
        return normalize(points, image_aspect=rgb.shape[1] / rgb.shape[0]), 'ok'

    def close(self):
        self.hands.close()
