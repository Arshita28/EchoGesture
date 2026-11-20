import numpy as np

def summarize(seq):
    seq = seq.reshape(32, 154)
    # Canonicalize the more frequently visible hand into the right slot.
    from asl_lab.word_invariant import mirrored
    if seq[:, 151].sum() > seq[:, 152].sum():
        seq = mirrored(seq)
    return np.concatenate([block.mean(axis=0) for block in np.array_split(seq, 4)] +
                          [seq.std(axis=0)])


def summarize_batch(samples):
    return np.stack([summarize(sample) for sample in samples])
