import numpy as np
from asl_lab.word_model import classify_word
from asl_lab.word_features import WORD_FEATURE_VERSION


def test_rejection_candidate_does_not_force_close_ambiguous_scores():
    class Model:
        classes_=np.array(['computer','who','deaf'])
        def decision_function(self,x):return np.array([[-.1,-.101,-.2]])
    features=np.zeros((32,154),np.float32);features[:,-3:]=1
    bundle=dict(model=Model(),feature_version=WORD_FEATURE_VERSION,
        rejection=dict(max_distance=.18,min_gap=.005))
    result=classify_word(bundle,features)
    assert result['status']=='uncertain_word' and result['label'] is None
    # Legacy model remains compatible when rejection is not configured.
    del bundle['rejection']
    assert classify_word(bundle,features)['label']=='computer'


def test_other_sign_class_never_becomes_a_displayed_word():
    class Model:
        classes_=np.array(['NONE','computer','who'])
        def decision_function(self,x):return np.array([[3.,1.,0.]])
    features=np.zeros((32,154),np.float32);features[:,-3:]=1
    result=classify_word(dict(model=Model(),feature_version=WORD_FEATURE_VERSION),features)
    assert result['status']=='uncertain_word' and result['label'] is None


def test_background_metrics_include_valid_word_rejections_as_errors():
    from scripts.train_word_background import metrics
    r=metrics(['computer','who','NONE','NONE'],['computer','NONE','NONE','who'])
    assert r['known_correct']==1 and r['known_total']==2 and r['known_rejected']==1
    assert r['other_rejected']==1 and r['other_total']==2 and r['balanced_utility']==.5


def test_other_sign_internal_class_is_not_a_word_alternative():
    class Model:
        classes_=np.array(['NONE','computer','who'])
        def decision_function(self,x):return np.array([[2.,3.,1.]])
    features=np.zeros((32,154),np.float32);features[:,-3:]=1
    result=classify_word(dict(model=Model(),feature_version=WORD_FEATURE_VERSION),features)
    assert result['label']=='computer'
    assert [r['label'] for r in result['candidates']]==['computer','who']
