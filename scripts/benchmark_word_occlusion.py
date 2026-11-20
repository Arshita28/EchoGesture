"""Compare a hand-dropout candidate and tracking rates on separate test footage."""
import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.base import clone
from asl_lab.word_dense import extract_dense
from asl_lab.word_features import WordExtractor
from asl_lab.word_svm_features import summarize
from scripts.compare_citizen_svm import word_dataset
from scripts.extract_words import quality


class MaskedExtractor(WordExtractor):
    def extract(self, rgb):
        h,w=rgb.shape[:2]
        rgb=rgb.copy()
        rgb[int(.35*h):int(.55*h),int(.45*w):int(.60*w)] = 0
        return super().extract(rgb)


def main():
    baseline=joblib.load('models/word20-citizen-svm.joblib')
    y,x,_=word_dataset('train')
    rng=np.random.default_rng(42)
    augmented=[]
    for sample in x:
        seq=sample.reshape(32,154).copy()
        side=int(seq[:,151:153].sum(axis=0).argmax())
        start=int(rng.integers(0,25))
        frames=seq[start:start+8]
        frames[:,side*63:(side+1)*63]=0
        frames[:,126+2*side:128+2*side]=0
        frames[:,151+side]=0
        augmented.append(summarize(seq))
    model=clone(baseline).fit(np.vstack(([summarize(s) for s in x],augmented)),np.tile(y,2))
    joblib.dump(model,'models/word20-dropout-candidate.joblib')
    # Fixed first test clip per exact gloss. Never selected by predicted score.
    import csv
    root=Path('datasets/raw/asl-citizen-selected')
    clips={}
    for row in csv.DictReader((root/'test.csv').open()):
        path=root/row['Video file']
        if path.exists():clips.setdefault(row['Gloss'].lower(),path)
    results=[]
    for masked,cls in [(False,WordExtractor),(True,MaskedExtractor)]:
        tracker=cls()
        try:
            for rate in (25,12):
                for label,path in clips.items():
                    start=time.perf_counter()
                    try:
                        seq=extract_dense(path,tracker,tracking_fps=rate)[0]
                        vector=[summarize(seq)] if quality(seq)['quality_pass'] else None
                    except ValueError:vector=None
                    ms=(time.perf_counter()-start)*1000
                    results.append(dict(label=label,clip=path.name,masked=masked,fps=rate,tracking_ms=ms,
                        baseline=baseline.predict(vector)[0] if vector else None,
                        candidate=model.predict(vector)[0] if vector else None))
                group=[r for r in results if r['masked']==masked and r['fps']==rate]
                print(masked,rate,len(group),*[sum(r[k]==r['label'] for r in group) for k in ['baseline','candidate']],round(np.median([r['tracking_ms'] for r in group])),flush=True)
        finally:tracker.close()
    Path('reports/word20-dropout-speed.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
