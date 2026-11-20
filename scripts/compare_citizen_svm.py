"""Compare temporal RBF classifiers on cached word clips using validation only."""
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from scripts.extract_words import quality
from scripts.train_word_background import load


from asl_lab.word_svm_features import summarize

def word_dataset(split):
    root = Path('datasets/raw/asl-citizen-selected')
    labels = json.loads(Path('config/word20.json').read_text())['labels']
    rows = json.loads(Path('datasets/preprocessed/word20-dense/extraction.json').read_text())['rows']
    samples = [(r['label'], load(r), 'wlasl') for r in rows if r['split']==split and r['quality_pass']]
    name = 'val' if split=='validation' else split
    for r in csv.DictReader((root/(name+'.csv')).open()):
        label = r['Gloss'].lower()
        path = (root/r['Video file']).with_suffix('.npy')
        if label in labels and path.exists():
            x = np.load(path)
            if quality(x)['quality_pass']:
                samples.append((label,x.reshape(-1),'citizen'))
    return tuple(np.array(a) for a in zip(*samples))


def main():
    y,x,_ = word_dataset('train')
    vy,vx,source = word_dataset('validation')
    sx = np.stack([summarize(a) for a in x])
    sv = np.stack([summarize(a) for a in vx])
    baseline = joblib.load('models/word20-citizen.joblib')['model']
    def scores(truth,pred,src):
        return {s:int(sum(pred[src==s]==truth[src==s])) for s in ('wlasl','citizen')}
    old = scores(vy,baseline.predict(vx),source)
    best = None
    for c in (1,10,100):
        for gamma in ('scale',.01,.05):
            model=SVC(C=c,gamma=gamma,class_weight='balanced').fit(sx,y)
            score=scores(vy,model.predict(sv),source)
            rank=(min(score[s]/old[s] for s in old),sum(score.values()))
            if best is None or rank>best[0]: best=(rank,model,score)
    _,model,score=best
    ty,tx,src=word_dataset('test')
    pred=model.predict(np.stack([summarize(a) for a in tx]))
    report=dict(validation_baseline=old,validation_candidate=score,test_candidate=scores(ty,pred,src),
                test_baseline=scores(ty,baseline.predict(tx),src),C=model.C,gamma=model.gamma)
    print(report)
    Path('reports/word20-citizen-svm.json').write_text(json.dumps(report,indent=2)+'\n')
    joblib.dump(model,'models/word20-citizen-svm.joblib')


if __name__=='__main__':main()
