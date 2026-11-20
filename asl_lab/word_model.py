"""Small-data word classifiers, separate from alphabet and J/Z models."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from .temporal_match import aligned_distance
from .word_features import WORD_FEATURE_DIM, WORD_FEATURE_VERSION


class WordTemplateClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, x, y):
        self.templates_ = np.asarray(x).reshape(-1, 32, WORD_FEATURE_DIM).copy()
        self.labels_ = np.asarray(y); self.classes_ = np.unique(y)
        return self

    def decision_function(self, x):
        weights = np.ones(WORD_FEATURE_DIM)
        weights[126:130] = 2
        weights[130:144] = .5
        results=[]
        for seq in np.asarray(x).reshape(-1,32,WORD_FEATURE_DIM):
            distances=np.array([aligned_distance(seq*weights,t*weights) for t in self.templates_])
            results.append([-float(distances[self.labels_==c].min()) for c in self.classes_])
        return np.asarray(results)

    def predict(self,x):
        return self.classes_[self.decision_function(x).argmax(axis=1)]


def classify_word(bundle, features):
    if bundle.get('feature_version') != WORD_FEATURE_VERSION:
        raise ValueError('Word model feature version mismatch')
    x=np.asarray(features,dtype=np.float32)
    if x.shape!=(32,WORD_FEATURE_DIM) or not np.isfinite(x).all():
        raise ValueError('Expected a finite 32-frame word feature sequence')
    masks=x[:,-3:]
    body=float(masks[:,2].mean()); hands=float((masks[:,:2].max(axis=1)>0).mean())
    quality=dict(body_coverage=body,any_hand_coverage=hands)
    if body<.7 or hands<.5:
        return dict(status='insufficient_detection',label=None,candidates=[],quality=quality)
    model=bundle['model'];scores=np.asarray(model.decision_function(x.reshape(1,-1)))[0]
    order=np.argsort(scores)[::-1]
    if str(model.classes_[order[0]])=='NONE':
        return dict(status='uncertain_word',label=None,candidates=[],quality=quality,
            reason='Other-sign class matched; no supported word emitted')
    rejection=bundle.get('rejection')
    if rejection is not None:
        distance=float(-scores[order[0]])
        gap=float(scores[order[0]]-scores[order[1]])
        if distance>rejection['max_distance'] or gap<rejection['min_gap']:
            return dict(status='uncertain_word',label=None,candidates=[],quality=quality,
                reason='No sufficiently distinct supported-word match',distance=distance,gap=gap)
    # Raw decision values rank known classes only. They are not probabilities.
    return dict(status='prototype_match',label=str(model.classes_[order[0]]),
        candidates=[dict(label=str(model.classes_[i]),decision_score=float(scores[i])) for i in order if str(model.classes_[i])!='NONE'][:3],
        quality=quality,scope=f'Known words only; unknown signs are not reliably rejected.')
