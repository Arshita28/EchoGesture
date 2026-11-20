"""Mask-aware temporal candidate. Original word prototype remains untouched."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from .word_features import WORD_FEATURE_DIM


def usable_sequence(sequence):
    """Trim leading/trailing undetected frames, preserving internal detection gaps."""
    seq=np.asarray(sequence,dtype=np.float32).reshape(-1,WORD_FEATURE_DIM)
    if not np.isfinite(seq).all():raise ValueError('Nonfinite sequence')
    visible=(seq[:,-1]>.5)&(seq[:,-3:-1].max(axis=1)>.5)
    indices=np.flatnonzero(visible)
    if len(indices)<4:return None
    return seq[indices[0]:indices[-1]+1]


def masked_distance(a,b,location_weight=1.):
    """DTW with no fake zero-hand comparisons; missing hands incur a penalty."""
    total=np.zeros((len(a),len(b)),dtype=float)
    count=np.zeros_like(total)
    for hand in (0,1):
        left=a[:,-3+hand]>.5;right=b[:,-3+hand]>.5
        both=left[:,None]&right[None,:];different=left[:,None]!=right[None,:]
        shape=np.mean((a[:,None,hand*63:(hand+1)*63]-b[None,:,hand*63:(hand+1)*63])**2,axis=2)
        location=np.mean((a[:,None,126+hand*2:128+hand*2]-b[None,:,126+hand*2:128+hand*2])**2,axis=2)
        total+=np.where(both,shape+location_weight*location,0)+different*.4
        count+=both|different
    cost=np.divide(total,count,out=np.full_like(total,.8),where=count>0)
    previous=np.full(len(b)+1,np.inf);previous[0]=0
    for row in cost:
        current=np.full(len(b)+1,np.inf)
        for j,value in enumerate(row,1):current[j]=value+min(current[j-1],previous[j],previous[j-1])
        previous=current
    return float(np.sqrt(previous[-1]/(len(a)+len(b))))


class MaskedWordClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,location_weight=1.):self.location_weight=location_weight
    def fit(self,x,y):
        self.templates_=[usable_sequence(s) for s in np.asarray(x).reshape(-1,32,WORD_FEATURE_DIM)]
        if any(s is None for s in self.templates_):raise ValueError('Training sequence lacks visible hands')
        self.labels_=np.asarray(y);self.classes_=np.unique(y)
        return self
    def decision_function(self,x):
        rows=[]
        for raw in np.asarray(x).reshape(-1,32,WORD_FEATURE_DIM):
            sequence=usable_sequence(raw)
            if sequence is None:rows.append(np.full(len(self.classes_),-1e6));continue
            distances=np.array([masked_distance(sequence,t,self.location_weight) for t in self.templates_])
            rows.append([-float(np.min(distances[self.labels_==label])) for label in self.classes_])
        return np.asarray(rows)
    def predict(self,x):return self.classes_[self.decision_function(x).argmax(axis=1)]
