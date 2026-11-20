import numpy as np
from asl_lab.word_candidate import usable_sequence,masked_distance,MaskedWordClassifier
from asl_lab.word_features import WORD_FEATURE_DIM


def fixture():
    rng=np.random.default_rng(1);x=rng.normal(0,.1,(16,WORD_FEATURE_DIM)).astype(np.float32)
    x[:,-3:]=1;return x


def test_trim_outer_missing_frames_preserves_internal_gaps():
    x=fixture();x[8]=0;padded=np.vstack([np.zeros((8,WORD_FEATURE_DIM)),x,np.zeros((8,WORD_FEATURE_DIM))])
    seq=usable_sequence(padded)
    assert len(seq)==16 and not seq[8].any()
    assert usable_sequence(np.zeros((32,WORD_FEATURE_DIM))) is None


def test_masked_distance_respects_hand_location_and_missing_hand_masks():
    x=fixture();assert masked_distance(x,x)==0
    moved=x.copy();moved[:,126:130]+=.5
    assert masked_distance(x,moved)>0
    absent=x.copy();absent[:,-3:-1]=0
    assert masked_distance(x,absent)>0


def test_padding_does_not_change_candidate_prediction():
    x=fixture();base=np.vstack([x,x]);other=base.copy();other[:,:126]+=2
    model=MaskedWordClassifier().fit([base.reshape(-1),other.reshape(-1)],['a','b'])
    padded=np.vstack([np.zeros((8,WORD_FEATURE_DIM)),x,np.zeros((8,WORD_FEATURE_DIM))])
    assert model.predict(padded.reshape(1,-1))[0]=='a'
