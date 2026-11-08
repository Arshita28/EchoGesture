"""Explicit finger geometry, derived from the same normalized landmarks."""
import numpy as np


def geometry_features(x):
    points = np.asarray(x).reshape(-1, 21, 3)
    # Pairwise distances capture fingertip separation and thumb/finger contact.
    a, b = np.triu_indices(21, 1)
    distances = np.linalg.norm(points[:, a] - points[:, b], axis=2)
    # Joint cosines distinguish extended and curled fingers without angle wrap.
    angles = []
    for chain in [(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)]:
        for i in range(1,4):
            u = points[:,chain[i-1]] - points[:,chain[i]]
            v = points[:,chain[i+1]] - points[:,chain[i]]
            angles.append((u*v).sum(axis=1) / np.maximum(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1),1e-12))
    return np.column_stack([points.reshape(-1,63), distances, *angles])
