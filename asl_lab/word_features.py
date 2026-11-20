"""Separate two-hand/body features for word clips; does not change alphabet features.

Image-space xyz is pseudo-3D, not calibrated metric 3D. Hand z origins differ
from pose z, so hand depth is kept wrist-relative, never subtracted from pose z.
"""
import numpy as np

WORD_FEATURE_VERSION = 'holistic-body-hands-v1'
POSE_INDICES = (0, 11, 12, 13, 14, 15, 16)
WORD_FEATURE_DIM = 154


def word_features(left, right, pose, aspect=1.0, visibility_threshold=0.5):
    if not np.isfinite(aspect) or aspect <= 0:
        raise ValueError('Expected positive width/height aspect')
    output = np.zeros(WORD_FEATURE_DIM, dtype=np.float32)
    if pose is None:
        return output
    pose = np.asarray(pose, dtype=float).copy()
    if pose.shape != (33, 4) or not np.isfinite(pose).all():
        raise ValueError('Expected 33 finite pose xyz/visibility entries')
    if np.any((pose[:, 3] < 0) | (pose[:, 3] > 1)):
        raise ValueError('Pose visibility must be between zero and one')
    pose[:, 1] /= aspect
    if min(pose[11, 3], pose[12, 3]) < visibility_threshold:
        return output
    center = (pose[11, :2] + pose[12, :2]) / 2
    axis = pose[11, :2] - pose[12, :2]
    scale = np.linalg.norm(axis)
    if scale < 1e-6:
        return output
    axis /= scale
    rotation = np.array([[axis[0], axis[1]], [-axis[1], axis[0]]])
    def body_xy(points):
        return ((points - center) @ rotation.T) / scale
    wrists = np.zeros((2, 2), dtype=float)
    masks = np.zeros(3, dtype=float); masks[2] = 1
    for slot, hand in enumerate((left, right)):
        if hand is None:
            continue
        hand = np.asarray(hand, dtype=float).copy()
        if hand.shape != (21, 3) or not np.isfinite(hand).all():
            raise ValueError('Expected 21 finite hand xyz landmarks')
        hand[:, 1] /= aspect
        wrist = hand[0, :2].copy()
        local = hand - hand[0]
        local[:, :2] = local[:, :2] @ rotation.T
        radius = np.linalg.norm(local, axis=1).max()
        if radius < 1e-6:
            continue
        output[slot*63:(slot+1)*63] = (local / radius).reshape(-1)
        wrists[slot] = body_xy(wrist)
        masks[slot] = 1
    output[126:130] = wrists.reshape(-1)
    selected = pose[list(POSE_INDICES)]
    valid = selected[:, 3] >= visibility_threshold
    body = body_xy(selected[:, :2]); body[~valid] = 0
    output[130:144] = body.reshape(-1)
    output[144:151] = selected[:, 3]
    output[151:] = masks
    return output


class WordExtractor:
    """Reset by constructing once per clip; actor-relative hand slots via Holistic."""
    def __init__(self):
        import mediapipe as mp
        self.model = mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=1, smooth_landmarks=False,
            enable_segmentation=False, refine_face_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def extract(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Expected uint8 RGB image')
        result = self.model.process(np.ascontiguousarray(rgb))
        def hand(value):
            return [[p.x, p.y, p.z] for p in value.landmark] if value else None
        pose = [[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks.landmark] if result.pose_landmarks else None
        return word_features(hand(result.left_hand_landmarks), hand(result.right_hand_landmarks),
                             pose, aspect=rgb.shape[1]/rgb.shape[0])

    def close(self):
        self.model.close()
