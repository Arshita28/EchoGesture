import numpy as np

from asl_lab.word_svm_features import summarize
from asl_lab.word_invariant import mirrored


def summarize_motion(seq):
    """
    Preserve the existing 770 features and add 16 wrist-displacement features.

    The 16 additional values represent:
    4 temporal sections x 2 hands x 2 movement axes.
    """
    seq = np.asarray(seq, dtype=np.float32).reshape(32, 154).copy()

    # Apply exactly the same hand canonicalisation as the existing model.
    if seq[:, 151].sum() > seq[:, 152].sum():
        seq = mirrored(seq)

    # Preserve all existing 770 features.
    existing = summarize(seq)

    movement = []

    # Four temporal sections, each containing eight frames.
    for block in np.array_split(seq, 4):
        for hand in range(2):
            wrist_start = 126 + hand * 2
            wrist = block[:, wrist_start:wrist_start + 2]

            # Only use movement between consecutive visible hand detections.
            visible = block[:, 151 + hand] > 0.5
            valid_pairs = visible[:-1] & visible[1:]

            if np.any(valid_pairs):
                displacement = np.diff(wrist, axis=0)
                total_displacement = displacement[valid_pairs].sum(axis=0)
            else:
                total_displacement = np.zeros(2, dtype=np.float32)

            movement.extend(total_displacement.tolist())

    return np.concatenate([
        existing,
        np.asarray(movement, dtype=np.float32)
    ])


def summarize_motion_batch(samples):
    return np.stack([summarize_motion(sample) for sample in samples])
