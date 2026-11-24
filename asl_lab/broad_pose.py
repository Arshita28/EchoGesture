"""Two-dimensional body and hand features shared by research and video inference."""
import numpy as np

BODY = [0, 5, 6, 7, 8, 9, 10]
HAND = list(range(91, 133))
MEDIAPIPE_BODY = [0, 11, 12, 13, 14, 15, 16]


def describe(points, aspect):
    """Input order: nose, shoulders, elbows, wrists, then 21 joints per hand."""
    points = np.asarray(points, dtype=np.float32).copy()
    visible = np.isfinite(points).all(axis=-1) & np.any(points != 0, axis=-1)
    points[:, :, 0] *= aspect
    shoulder_span = np.linalg.norm(points[:, 1] - points[:, 2], axis=1)
    tracking = visible[:, 1] & visible[:, 2] & (shoulder_span > .02)
    hand_seen = visible[:, 7:28].sum(axis=1) >= 10
    hand_seen |= visible[:, 28:].sum(axis=1) >= 10
    indices = np.flatnonzero(tracking & hand_seen)
    if len(indices) < 3:
        raise ValueError('Too few frames with shoulders and hand landmarks')
    idx = indices[np.rint(np.linspace(0, len(indices) - 1, 32)).astype(int)]
    p, mask = points[idx], visible[idx]
    center = (p[:, 1] + p[:, 2]) * .5
    body = (p - center[:, None]) / shoulder_span[idx, None, None]
    blocks = [body[:, :7], body[:, [7, 28]]]
    for start in (7, 28):
        hand = p[:, start:start + 21] - p[:, start:start + 1]
        valid = mask[:, start:start + 21] & mask[:, start:start + 1]
        hand[~valid] = 0
        radius = np.linalg.norm(hand, axis=2).max(axis=1)
        blocks.append(hand / np.maximum(radius[:, None, None], .005))
    blocks[0][~mask[:, :7]] = 0
    blocks[1][~mask[:, [7, 28]]] = 0
    seq = np.concatenate(blocks, axis=1).reshape(32, -1)
    seq = np.clip(seq, -4, 4)
    return np.concatenate([b.mean(axis=0) for b in np.array_split(seq, 6)] + [seq.std(axis=0)])


def describe_video(path):
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        with mp.solutions.holistic.Holistic(model_complexity=1, smooth_landmarks=False) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                result = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                points = np.zeros((49, 2), dtype=np.float32)
                if result.pose_landmarks:
                    for j, k in enumerate(MEDIAPIPE_BODY):
                        joint = result.pose_landmarks.landmark[k]
                        if joint.visibility >= .5:
                            points[j] = joint.x, joint.y
                for start, hand in ((7, result.left_hand_landmarks), (28, result.right_hand_landmarks)):
                    if hand:
                        points[start:start + 21] = [(v.x, v.y) for v in hand.landmark]
                frames.append(points)
    finally:
        cap.release()
    if not frames:
        raise ValueError('No decoded frames')
    return describe(frames, width / height)


def from_cached(seq):
    # Express canonical hand shape at its body-relative wrist. Its arbitrary
    # scale cancels in describe(), which separately encodes hand and wrist.
    body = seq[:, 130:144].reshape(-1, 7, 2).copy()
    body[seq[:, 144:151] < .5] = 0
    hands = []
    for slot in range(2):
        hand = seq[:, slot * 63:(slot + 1) * 63].reshape(-1, 21, 3)[:, :, :2]
        wrist = seq[:, 126 + 2 * slot:128 + 2 * slot]
        hand = .25 * hand + wrist[:, None]
        hand[seq[:, 151 + slot] <= 0] = 0
        hands.append(hand)
    return describe(np.concatenate([body, *hands], axis=1), 1)
