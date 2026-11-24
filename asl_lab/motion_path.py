"""Conservative path evidence for J/Z, complementing the learned templates."""
import numpy as np


def simplify(path, tolerance=.10):
    if len(path) <= 2:
        return path
    line = path[-1] - path[0]
    length = np.linalg.norm(line)
    if length < 1e-6:
        distances = np.linalg.norm(path - path[0], axis=1)
    else:
        distances = np.abs(line[0] * (path[:, 1]-path[0, 1]) - line[1] * (path[:, 0]-path[0, 0])) / length
    index = int(np.argmax(distances))
    if distances[index] <= tolerance or index in (0, len(path)-1):
        return path[[0, -1]]
    return np.concatenate([simplify(path[:index+1], tolerance)[:-1], simplify(path[index:], tolerance)])


def hook_turn(path):
    """Require a downward stroke followed by a lateral or rising hook."""
    delta=np.diff(simplify(path),axis=0)
    for first,second in zip(delta,delta[1:]):
        descending=first[1]>.25 and abs(first[0])<.8*first[1]
        turning=abs(second[0])>.15 and second[1]<.6*abs(second[0])
        if descending and turning:
            return True
    return False


def path_evidence(frames):
    points = np.array([f['points'] for f in frames], dtype=float)
    points[:, :, 1] /= frames[0]['aspect']
    radius = np.linalg.norm(points - points[:, :1], axis=2)
    scale = np.median(radius.max(axis=1))
    if scale < 1e-6:
        return {'label': None}
    ratios = np.median(radius[:, [8, 12, 16, 20]] / np.maximum(radius[:, [6, 10, 14, 18]], 1e-6), axis=0)
    # The index of Z may tilt toward the camera. Compare it to the folded fingers.
    index_shape = ratios[0] > .95 and max(ratios[1:]) < 1.05 and ratios[0]-max(ratios[1:]) > .18
    little_shape = ratios[3] > 1.15 and max(ratios[:3]) < 1.05
    j_curve=False
    for label, tip, shape in [('Z', 8, index_shape), ('J', 20, little_shape)]:
        if not shape:
            continue
        path = points[:, tip, :2] / scale
        path = np.stack([np.median(path[max(0,i-1):i+2], axis=0) for i in range(len(path))])
        path = simplify(path)
        delta = np.diff(path, axis=0)
        if label == 'Z':
            # Locate the actual trace within hand entry and exit movements.
            for start in range(len(delta)-2):
                strokes=delta[start:start+3]
                dx,dy=strokes[:,0],strokes[:,1]
                zigzag=min(abs(dx))>.35 and dx[0]*dx[1]<0 and dx[1]*dx[2]<0
                downward=abs(dx[1])>1e-6 and dy[1]>.25 and abs(dy[1]/dx[1])>.4
                horizontal=abs(dy[0])<abs(dx[0])*.7 and abs(dy[2])<abs(dx[2])*.7
                if zigzag and downward and horizontal:
                    return {'label':'Z','reason':'Index handshape and three-stroke zigzag'}
        if label == 'J':
            j_curve=hook_turn(path)
            if j_curve:
                return {'label':'J','reason':'Little-finger handshape and completed downward hook'}
    eligible = []
    for label, tip, shape in [('J',20,little_shape),('Z',8,index_shape)]:
        if shape and (label!='J' or j_curve) and np.linalg.norm(np.ptp(points[:,tip,:2],axis=0))/scale > .3:
            eligible.append(label)
    return {'label': None, 'eligible': eligible}
