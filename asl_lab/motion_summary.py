"""Sequence features for a compact J/Z/background classifier."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MotionSummary(TransformerMixin, BaseEstimator):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        vectors=[]
        for sample in np.asarray(x).reshape(-1,24,21,3):
            radius=np.linalg.norm(sample-sample[:,:1],axis=2)
            scale=max(float(np.median(radius.max(axis=1))),1e-6)
            ratios=np.clip(radius[:,[8,12,16,20]]/np.maximum(radius[:,[6,10,14,18]],1e-6),0,3)
            paths=sample[:,[0,8,20],:2]/scale
            paths-=paths[:1].copy()
            velocity=np.diff(paths,axis=0,prepend=paths[:1])
            # Retain signed travel and its direction changes across the gesture.
            vector=np.concatenate([paths.reshape(-1),velocity.reshape(-1),
                                   ratios.mean(axis=0),ratios.std(axis=0),
                                   np.ptp(paths,axis=0).reshape(-1)])
            vectors.append(vector)
        return np.stack(vectors)


class MotionZSpecialist(BaseEstimator):
    """Preserve accepted reference gestures and consult the trained SVM on uncertainty."""
    def __init__(self, reference, specialist):
        self.reference=reference
        self.specialist=specialist
        self.classes_=reference.classes_
        if not np.array_equal(self.classes_,specialist.classes_):
            raise ValueError('Motion class order mismatch')

    def predict_proba(self,x):
        original=self.reference.predict_proba(x)
        extra=self.specialist.predict_proba(x)
        for i in range(len(original)):
            first=self.classes_[np.argmax(original[i])]
            second=self.classes_[np.argmax(extra[i])]
            confirmed=first!='NONE' and original[i].max()>=.7
            if not confirmed and second in ('J','Z'):
                original[i]=extra[i]
        return original

    def predict(self,x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]
