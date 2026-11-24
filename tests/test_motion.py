import numpy as np
import pytest
from asl_lab.motion import motion_features


def sequence(direction=1):
    base=np.random.default_rng(4).uniform(.1,.8,(21,3))
    return [{'points':(base+[direction*i*.01,0,0]).tolist(),'time':i*150,'aspect':1.0} for i in range(10)]


def test_motion_retains_trajectory_and_normalizes_global_scale():
    frames=sequence()
    expected=motion_features(frames)
    assert expected.shape==(24*63,)
    assert np.isfinite(expected).all()
    transformed=[{**f,'points':(np.array(f['points'])*2+[3,4,5]).tolist()} for f in frames]
    np.testing.assert_allclose(expected,motion_features(transformed),atol=1e-12)
    assert not np.allclose(expected,motion_features(sequence(-1)))


def test_motion_rejects_incomplete_and_invalid_sequences():
    with pytest.raises(ValueError):motion_features(sequence()[:2])
    with pytest.raises(ValueError):motion_features([{}]*10)
    frames=sequence();frames[4]['time']=frames[3]['time']
    with pytest.raises(ValueError):motion_features(frames)


def test_motion_capture_and_untrained_inference(tmp_path,monkeypatch):
    import app
    monkeypatch.setattr(app,'ROOT',tmp_path)
    client=app.create_app().test_client()
    payload={'frames':sequence(),'action':'predict'}
    assert client.post('/api/motion',json=payload).json['status']=='untrained'
    payload.update(action='collect',label='J',group='signer_01',split='train')
    assert client.post('/api/motion',json=payload).json['status']=='saved'
    payload['split']='test'
    assert client.post('/api/motion',json=payload).status_code==400
    assert client.post('/api/motion',json={'frames':[{}]*10}).status_code==400


def test_motion_training_uses_partitioned_sequences(tmp_path,monkeypatch):
    import json
    from scripts.train_motion import train
    monkeypatch.chdir(tmp_path)
    directory=tmp_path/'data/processed/motion';directory.mkdir(parents=True)
    (tmp_path/'models').mkdir()
    # Synthetic infrastructure fixtures only; no recognition result is claimed.
    for split in ('train','validation','test'):
        for label,direction in [('J',1),('Z',-1),('NONE',0)]:
            for i in range(2):
                frames=sequence(direction)
                for f in frames:
                    f['points']=(np.array(f['points'])+[i*.001,0,0]).tolist()
                row={'label':label,'group':split,'split':split,'frames':frames}
                (directory/f'{split}-{label}-{i}.json').write_text(json.dumps(row))
    # Identical sequences crossing partitions must be rejected.
    with pytest.raises(ValueError,match='crosses'):train(directory)
    for path in directory.glob('*.json'):
        row=json.loads(path.read_text())
        offset={'train':0,'validation':.003,'test':.007}[row['split']]
        for f in row['frames']:f['points']=(np.array(f['points'])+[offset,0,0]).tolist()
        path.write_text(json.dumps(row))
    result=train(directory)
    assert set(result['counts'])=={'train','validation','test'}
    assert (tmp_path/'models/motion.joblib').exists()


def test_time_alignment_tolerates_repeated_pause_samples():
    from asl_lab.temporal_match import aligned_distance,TemporalMatcher
    a=np.array([[0.],[1.],[2.]])
    b=np.array([[0.],[0.],[1.],[1.],[2.],[2.]])
    assert aligned_distance(a,b)==0
    assert aligned_distance(a,b+3)>0
    x=np.array([motion_features(sequence(i)) for i in [-1,0,1]])
    matcher=TemporalMatcher().fit(x,['Z','NONE','J'])
    assert list(matcher.predict(x))==['Z','NONE','J']
    np.testing.assert_allclose(matcher.predict_proba(x).sum(axis=1),1)


def test_short_complete_motion_keeps_same_feature_shape():
    frames=sequence()[:6]
    for i,frame in enumerate(frames):frame['time']=i*75
    assert motion_features(frames).shape==(24*63,)
    for i,frame in enumerate(frames):frame['time']=i*30
    with pytest.raises(ValueError):motion_features(frames)


def test_motion_specialist_preserves_confirmed_reference_and_recovers_uncertainty():
    from asl_lab.motion_summary import MotionZSpecialist
    class Fixed:
        classes_=np.array(['J','NONE','Z'])
        def __init__(self,scores):self.scores=np.array(scores)
        def predict_proba(self,x):return self.scores.copy()
    reference=Fixed([[.9,.05,.05],[.2,.6,.2],[.2,.6,.2]])
    specialist=Fixed([[.02,.03,.95],[.05,.05,.9],[.05,.9,.05]])
    model=MotionZSpecialist(reference,specialist)
    scores=model.predict_proba(np.zeros((3,1512)))
    np.testing.assert_allclose(scores[0],reference.scores[0])
    np.testing.assert_allclose(scores[1],specialist.scores[1])
    np.testing.assert_allclose(scores[2],reference.scores[2])



def test_motion_summary_translation_invariance_and_finiteness():
    from asl_lab.motion_summary import MotionSummary
    from asl_lab.motion import motion_features
    x=motion_features(sequence()).reshape(1,-1)
    transform=MotionSummary()
    features=transform.transform(x)
    assert features.shape==(1,302)
    assert np.isfinite(features).all()
    shifted=x.reshape(1,24,21,3)+[2.,3.,1.]
    np.testing.assert_allclose(features,transform.transform(shifted.reshape(1,-1)),atol=1e-10)
