import numpy as np
from asl_lab.motion_path import path_evidence


def sequence(label, moving=True):
    hand = np.zeros((21,3))
    # Folded fingers have tips closer to the wrist than their middle joints.
    for base in (5,9,13,17):
        hand[base:base+4,0] = np.linspace(-.15,.15,4) + (base-11)*.015
        hand[base:base+4,1] = [.3,.55,.4,.3]
    extended = 17 if label=='J' else 5
    hand[extended:extended+4,1] = [.3,.55,.8,1.]
    corners = np.array([[0,0],[.8,0],[0,.5],[.8,.5]]) if label=='Z' else np.array([[0,0],[0,.8],[.35,.8],[.5,.5]])
    positions = np.concatenate([np.linspace(a,b,10,endpoint=False) for a,b in zip(corners,corners[1:])] + [corners[-1:]])
    frames=[]
    for i,pos in enumerate(positions):
        points=hand.copy()
        if moving: points[:,:2] += pos
        points[:,:2] += [1.,1.]
        frames.append(dict(points=points.tolist(),time=i*50,aspect=1))
    return frames


def test_completed_z_and_mirror():
    frames=sequence('Z')
    assert path_evidence(frames)['label']=='Z'
    for f in frames:
        for p in f['points']:p[0]*=-1
    assert path_evidence(frames)['label']=='Z'


def test_j_hook_and_stationary_i():
    assert path_evidence(sequence('J'))['label']=='J'
    stationary=path_evidence(sequence('J',False))
    assert stationary['label'] is None
    assert 'J' not in stationary.get('eligible',[])


def test_straight_i_repositioning_cannot_confirm_j():
    for dx,dy in [(0,0),(1,0),(-1,0),(0,1),(0,-1),(.8,1),(-.8,-1)]:
        frames=sequence('J',moving=False)
        for i,frame in enumerate(frames):
            travel=min(i/15,1)*.6
            for point in frame['points']:
                point[0]+=travel*dx
                point[1]+=travel*dy
        evidence=path_evidence(frames)
        assert evidence['label']!='J'
        assert 'J' not in evidence.get('eligible',[])


def test_endpoint_rejects_i_even_with_high_j_model_score(tmp_path,monkeypatch):
    import app
    import joblib
    class AlwaysJ:
        classes_=np.array(['J','NONE','Z'])
        def predict_proba(self,x):return np.array([[.99,.005,.005]])
    monkeypatch.setattr(app,'ROOT',tmp_path)
    (tmp_path/'models').mkdir()
    (tmp_path/'models/motion.joblib').touch()
    from asl_lab.motion import MOTION_VERSION
    monkeypatch.setattr(joblib,'load',lambda path:{'model':AlwaysJ(),'version':MOTION_VERSION,'threshold':.7})
    client=app.create_app().test_client()
    frames=sequence('J',False)
    for i,f in enumerate(frames):
        for p in f['points']:p[1]-=min(i/15,1)*.6
    response=client.post('/api/motion',json={'action':'predict','frames':frames})
    assert response.status_code==200
    assert response.json['label'] is None
    assert response.json['candidate']=='NONE'
    valid=client.post('/api/motion',json={'action':'predict','frames':sequence('J')})
    assert valid.json['label']=='J'

def test_large_i_sweep_recognized_in_either_direction():
    for direction in (-1, 1):
        frames = sequence('J', moving=False)
        for index, frame in enumerate(frames):
            for point in frame['points']:
                point[0] += direction * 1.4 * min(index / 15, 1)
        assert path_evidence(frames)['label'] == 'J'
