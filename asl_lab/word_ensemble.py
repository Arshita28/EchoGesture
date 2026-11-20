"""Probability-ranked estimator adapter for the existing word inference contract."""
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin,clone


class RankedClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,estimator):self.estimator=estimator
    def fit(self,x,y):
        self.estimator_=clone(self.estimator).fit(x,y);self.classes_=self.estimator_.classes_
        return self
    def decision_function(self,x):
        # Log probabilities provide ordering, not a calibrated signing score.
        return np.log(np.maximum(self.estimator_.predict_proba(x),1e-12))
    def predict(self,x):return self.classes_[self.decision_function(x).argmax(axis=1)]
