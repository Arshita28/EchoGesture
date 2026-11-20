"""Experimental handedness-aware temporal matching with optional velocity cues."""
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from .word_model import WordTemplateClassifier
from .temporal_match import aligned_distance


def mirrored(sequence):
    s=np.asarray(sequence).reshape(32,154);out=s.copy()
    for target,source in [(0,1),(1,0)]:
        h=s[:,source*63:(source+1)*63].reshape(32,21,3).copy();h[:,:,0]*=-1
        out[:,target*63:(target+1)*63]=h.reshape(32,63)
        wrist=s[:,126+source*2:128+source*2].copy();wrist[:,0]*=-1
        out[:,126+target*2:128+target*2]=wrist
        out[:,151+target]=s[:,151+source]
    pose=s[:,130:144].reshape(32,7,2)[:,[0,2,1,4,3,6,5],:].copy();pose[:,:,0]*=-1
    out[:,130:144]=pose.reshape(32,14)
    out[:,144:151]=s[:,144:151][:,[0,2,1,4,3,6,5]]
    return out


class InvariantWordMatcher(ClassifierMixin,BaseEstimator):
    def __init__(self,mirror=False,velocity_weight=0.):
        self.mirror=mirror;self.velocity_weight=velocity_weight
    def encode(self,seq):
        weights=np.ones(154);weights[126:130]=2;weights[130:144]=.5
        seq=seq*weights
        if self.velocity_weight:
            velocity=np.vstack([np.zeros((1,154)),np.diff(seq,axis=0)])
            seq=np.concatenate([seq,velocity*self.velocity_weight],axis=1)
        return seq
    def fit(self,x,y):
        self.templates_=np.asarray(x).reshape(-1,32,154).copy();self.labels_=np.asarray(y);self.classes_=np.unique(y)
        self.encoded_=[self.encode(t) for t in self.templates_]
        return self
    def decision_function(self,x):
        outputs=[]
        for seq in np.asarray(x).reshape(-1,32,154):
            views=[self.encode(seq)]
            if self.mirror:views.append(self.encode(mirrored(seq)))
            distances=np.array([min(aligned_distance(view,t) for view in views) for t in self.encoded_])
            outputs.append([-float(distances[self.labels_==label].min()) for label in self.classes_])
        return np.asarray(outputs)
    def predict(self,x):return self.classes_[self.decision_function(x).argmax(axis=1)]
