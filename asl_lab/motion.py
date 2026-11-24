"""Temporal landmark features retain motion relative to the first wrist."""
import numpy as np

MOTION_VERSION = 'resampled-wrist-origin-v1'
MOTION_LABELS = ('J', 'Z', 'NONE')


def motion_features(frames, steps=24):
    if not isinstance(frames, list) or not 6 <= len(frames) <= 120:
        raise ValueError('Record 6–120 detected hand frames')
    try:
        points = np.array([f['points'] for f in frames], dtype=float)
        times = np.array([f['time'] for f in frames], dtype=float)
        aspects = np.array([f['aspect'] for f in frames], dtype=float)
    except (KeyError, TypeError, ValueError):
        raise ValueError('Invalid sequence frame data') from None
    if points.shape != (len(frames),21,3) or not np.isfinite(points).all():
        raise ValueError('Invalid sequence landmarks')
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or np.any(np.diff(times) > 750) or times[-1]-times[0] < 350:
        raise ValueError('Sequence must cover at least 0.35 seconds with increasing timestamps')
    if not np.isfinite(aspects).all() or np.any(aspects <= 0) or np.ptp(aspects) > .01:
        raise ValueError('Keep frame aspect ratio constant while recording')
    points[:,:,1] /= aspects[:,None]
    scale = np.linalg.norm(points[0]-points[0,0],axis=1).max()
    if scale < 1e-6:
        raise ValueError('Degenerate initial hand')
    # Unlike static features, do not recenter each frame: this preserves the path.
    points = (points-points[0,0]) / scale
    target = np.linspace(times[0],times[-1],steps)
    flat = points.reshape(len(frames),63)
    return np.stack([np.interp(target,times,flat[:,i]) for i in range(63)],axis=1).reshape(-1)
