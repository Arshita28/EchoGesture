"""Nearest-template temporal matching with dynamic time alignment."""
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin


def aligned_distance(a,b):
    cost=np.mean((a[:,None,:]-b[None,:,:])**2,axis=2)
    previous=np.full(len(b)+1,np.inf);previous[0]=0
    for row in cost:
        current=np.full(len(b)+1,np.inf)
        for j,value in enumerate(row,1):current[j]=value+min(current[j-1],previous[j],previous[j-1])
        previous=current
    return float(np.sqrt(previous[-1]/(len(a)+len(b))))


class TemporalMatcher(ClassifierMixin,BaseEstimator):
    def __init__(self,temperature=.03):self.temperature=temperature
    def fit(self,x,y):
        self.templates_=np.asarray(x).reshape(-1,24,63)
        self.labels_=np.asarray(y);self.classes_=np.unique(y)
        return self
    def predict_proba(self,x):
        probabilities=[]
        for seq in np.asarray(x).reshape(-1,24,63):
            distances=np.array([aligned_distance(seq,t) for t in self.templates_])
            # Equal class influence despite more source background templates.
            class_distances=np.array([np.min(distances[self.labels_==c]) for c in self.classes_])
            scores=np.exp(-(class_distances-class_distances.min())/self.temperature)
            probabilities.append(scores/scores.sum())
        return np.array(probabilities)
    def predict(self,x):return self.classes_[self.predict_proba(x).argmax(axis=1)]
