import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from asl_lab.word_invariant import mirrored,InvariantWordMatcher
from asl_lab.word_ensemble import RankedClassifier


def test_mirror_twice_preserves_features_and_swaps_masks():
    x=np.random.default_rng(1).normal(size=(32,154));x[:,151:]=[1,0,1]
    y=mirrored(x);assert (y[:,151:]==[0,1,1]).all()
    np.testing.assert_allclose(mirrored(y),x)


def test_ranked_classifier_preserves_probability_order():
    x=np.array([[0],[.1],[1],[1.1]]);y=['a','a','b','b']
    model=RankedClassifier(ExtraTreesClassifier(n_estimators=5,random_state=1)).fit(x,y)
    assert list(model.predict(x))==y
    assert np.isfinite(model.decision_function(x)).all()


def test_invariant_matcher_self_distance_and_shape():
    x=np.random.default_rng(2).normal(size=(2,32*154))
    model=InvariantWordMatcher(False,1).fit(x,['a','b'])
    assert list(model.predict(x))==['a','b']
