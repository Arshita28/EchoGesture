"""Temporal pyramid features for small-data word recognition experiments."""
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from .word_features import WORD_FEATURE_DIM


class WordSummary(TransformerMixin,BaseEstimator):
    def fit(self,x,y=None):return self
    def transform(self,x):
        results=[]
        for seq in np.asarray(x).reshape(-1,32,WORD_FEATURE_DIM):
            parts=[]
            # Summaries retain order at multiple scales while avoiding literal
            # alignment of each observed frame with one training timestamp.
            for bins in [1,2,4]:
                for chunk in np.array_split(seq,bins):
                    for hand in [0,1]:
                        valid=chunk[:,-3+hand]>.5
                        values=np.concatenate([chunk[:,hand*63:(hand+1)*63],chunk[:,126+hand*2:128+hand*2]],axis=1)[valid]
                        if len(values):
                            parts.extend([values.mean(axis=0),values.std(axis=0),values[-1]-values[0]])
                        else:parts.extend([np.zeros(65)]*3)
                    parts.append(chunk[:,-3:].mean(axis=0))
            results.append(np.concatenate(parts))
        return np.asarray(results)
