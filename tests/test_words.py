"""Prototype infrastructure checks; synthetic arrays are not ASL accuracy evidence."""
import io
import json
import joblib
import numpy as np
import pytest
from asl_lab.word_features import WORD_FEATURE_DIM,WORD_FEATURE_VERSION
from asl_lab.word_model import WordTemplateClassifier,classify_word
from app import create_app


def fixture_bundle():
    rng=np.random.default_rng(3)
    x=[];y=[]
    for label,offset in [('one',-.5),('two',0),('three',.5)]:
        seq=rng.normal(offset,.02,(32,WORD_FEATURE_DIM)).astype(np.float32);seq[:,-3:]=1
        x.append(seq.reshape(-1));y.append(label)
    model=WordTemplateClassifier().fit(x,y)
    return dict(model=model,labels=y,name='fixture',feature_version=WORD_FEATURE_VERSION),np.array(x)


def test_word_model_returns_ranked_known_classes_and_rejects_missing_detection():
    bundle,x=fixture_bundle()
    result=classify_word(bundle,x[0].reshape(32,WORD_FEATURE_DIM))
    assert result['label']=='one' and result['status']=='prototype_match'
    assert len(result['candidates'])==3
    blank=classify_word(bundle,np.zeros((32,WORD_FEATURE_DIM)))
    assert blank['status']=='insufficient_detection' and blank['label'] is None
    with pytest.raises(ValueError):classify_word(bundle,np.zeros((2,WORD_FEATURE_DIM)))
    with pytest.raises(ValueError):classify_word({**bundle,'feature_version':'old'},x[0].reshape(32,WORD_FEATURE_DIM))


def test_word_routes_validate_inputs_delete_temporary_clips_and_keep_alphabet(monkeypatch,tmp_path):
    bundle,_=fixture_bundle();(tmp_path/'models').mkdir()
    joblib.dump(bundle,tmp_path/'models/words-prototype.joblib')
    monkeypatch.setattr('app.ROOT',tmp_path)
    calls=[]
    def fake_predict(path,bundle):
        calls.append(path);assert path.read_bytes()==b'fixture video'
        return dict(status='prototype_match',label='one')
    monkeypatch.setattr('asl_lab.word_routes.predict',fake_predict)
    client=create_app().test_client()
    assert client.get('/words').status_code==200
    assert client.get('/sign').status_code==200
    assert client.get('/api/words/status').json['words']==bundle['labels']
    assert client.post('/api/words/predict').status_code==400
    assert client.post('/api/words/predict',data={'video':(io.BytesIO(b'bad'),'bad.html')}).status_code==400
    response=client.post('/api/words/predict',data={'video':(io.BytesIO(b'fixture video'),'sign.webm')})
    assert response.json['label']=='one' and not calls[0].exists()
    assert client.get('/api/words/example/unknown').status_code==404


def test_word_examples_use_only_training_partition(monkeypatch,tmp_path):
    bundle,_=fixture_bundle();(tmp_path/'models').mkdir();joblib.dump(bundle,tmp_path/'models/words-prototype.joblib')
    folder=tmp_path/'datasets/preprocessed/words';folder.mkdir(parents=True)
    (folder/'quality-manifest.json').write_text(json.dumps([
        dict(label='one',split='test',video_id='1'),dict(label='one',split='train',video_id='2')]))
    videos=tmp_path/'datasets/raw/wlasl/videos';videos.mkdir(parents=True)
    (videos/'1.mp4').write_bytes(b'test');(videos/'2.mp4').write_bytes(b'train')
    monkeypatch.setattr('app.ROOT',tmp_path)
    response=create_app().test_client().get('/api/words/example/one')
    assert response.data==b'train'


def test_word_rejection_returns_no_label_when_distance_is_too_large():
    bundle,x=fixture_bundle()
    # Exact self match stays valid under a permissive calibrated policy.
    policy={**bundle,'rejection':{'max_distance':.1,'min_gap':0}}
    assert classify_word(policy,x[0].reshape(32,WORD_FEATURE_DIM))['label']=='one'
    moved=x[0].reshape(32,WORD_FEATURE_DIM).copy();moved[:,:151]+=10
    result=classify_word(policy,moved)
    assert result['status']=='uncertain_word' and result['label'] is None
    assert result['candidates']==[]
