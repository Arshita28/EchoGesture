import numpy as np
import pytest
from asl_lab.word_dense import crop_sequence
from asl_lab.word_features import WORD_FEATURE_DIM


def test_dense_cropping_ignores_outer_missing_detection_but_keeps_internal_gaps():
    x=np.zeros((100,WORD_FEATURE_DIM),np.float32);x[20:80,-3:]=1;x[49:52]=0
    y,t,info=crop_sequence(x,np.arange(100)/25)
    assert y.shape==(32,WORD_FEATURE_DIM)
    assert info['start_frame']==20 and info['end_frame_exclusive']==80
    assert t[0]==20/25 and t[-1]==79/25
    assert (y[:,-1]==0).any()


def test_dense_rejects_blank_or_invalid_timeline():
    x=np.zeros((32,WORD_FEATURE_DIM),np.float32)
    with pytest.raises(ValueError,match='eight'):crop_sequence(x,np.arange(32))
    with pytest.raises(ValueError):crop_sequence(x,np.zeros(32))


def test_summary_is_finite_and_retains_movement_direction():
    from asl_lab.word_summary import WordSummary
    x=np.zeros((32,WORD_FEATURE_DIM),np.float32);x[:,-3:]=1
    x[:,126]=np.linspace(0,1,32)
    result=WordSummary().fit_transform([x.reshape(-1),x[::-1].reshape(-1)])
    assert result.shape[0]==2 and np.isfinite(result).all()
    assert not np.allclose(result[0],result[1])


def test_word_inference_dispatches_dense_and_rejects_unknown_preprocessing(monkeypatch):
    import asl_lab.word_inference as module
    from asl_lab.word_dense import DENSE_VERSION
    calls=[]
    class Extractor:
        def close(self):calls.append('closed')
    monkeypatch.setattr(module,'WordExtractor',Extractor)
    monkeypatch.setattr(module,'inspect_video',lambda *a,**k:{'duration_s':3})
    def dense(*a):
        calls.append('dense');return np.zeros((32,WORD_FEATURE_DIM)),np.arange(32),{}
    monkeypatch.setattr(module,'extract_dense',dense)
    monkeypatch.setattr(module,'classify_word',lambda *a:{'status':'insufficient_detection','label':None})
    result=module.predict('fixture',bundle={'preprocessing':DENSE_VERSION,'name':'fixture','labels':[]})
    assert calls==['dense','closed'] and result['preprocessing']==DENSE_VERSION
    with pytest.raises(ValueError,match='Unknown'):module.predict('fixture',bundle={'preprocessing':'bad'})
    assert calls[-1]=='closed'
