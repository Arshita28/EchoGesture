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


class MotionTrajectoryMatcher(TemporalMatcher):
    """Match fingertip travel and handshape ratios across direction and tempo."""
    def encode(self, sequence, mirror=False):
        p=np.asarray(sequence).reshape(24,21,3).copy()
        if mirror:p[:,:,0]*=-1
        radius=np.linalg.norm(p-p[:,:1],axis=2)
        scale=max(float(np.median(radius.max(axis=1))),1e-6)
        shape=radius[:,[8,12,16,20]]/np.maximum(radius[:,[6,10,14,18]],1e-6)
        shape=np.clip(shape,0,3)
        index=(shape[:,0]>.95)&(shape[:,1:].max(axis=1)<1.05)
        little=(shape[:,3]>1.15)&(shape[:,:3].max(axis=1)<1.05)
        active=np.flatnonzero(index|little)
        if len(active)>=4:
            target=np.linspace(active[0],active[-1],24)
            source=np.arange(24)
            flat=p.reshape(24,-1)
            p=np.stack([np.interp(target,source,flat[:,i]) for i in range(63)],axis=1).reshape(24,21,3)
            shape=np.stack([np.interp(target,source,shape[:,i]) for i in range(4)],axis=1)
        paths=p[:,[0,8,20],:2]
        paths=(paths-paths[:1])/scale
        excursion=max(float(np.linalg.norm(np.ptp(paths[:,1:],axis=0),axis=1).max()),.3)
        paths/=excursion
        return np.concatenate([paths.reshape(24,6),shape*.6],axis=1)

    def fit(self,x,y):
        super().fit(x,y)
        self.paths_=[self.encode(t) for t in self.templates_]
        return self

    def predict_proba(self,x):
        probabilities=[]
        for seq in np.asarray(x).reshape(-1,24,63):
            views=[self.encode(seq),self.encode(seq,mirror=True)]
            distances=np.array([min(aligned_distance(v,t) for v in views) for t in self.paths_])
            nearest=np.array([distances[self.labels_==c].min() for c in self.classes_])
            scores=np.exp(-(nearest-nearest.min())/self.temperature)
            probabilities.append(scores/scores.sum())
        return np.asarray(probabilities)
