"""Fit only real collected sequences; select on explicit validation groups."""
import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score,classification_report
from asl_lab.motion import motion_features,MOTION_LABELS,MOTION_VERSION


def train(directory=Path('data/processed/motion')):
    rows=[json.loads(p.read_text()) for p in directory.glob('*.json')]
    if not rows:raise ValueError('Collect real motion sequences first')
    groups={};digests={}
    for r in rows:
        if r['label'] not in MOTION_LABELS or r['split'] not in ('train','validation','test') or not r['group']:raise ValueError('Invalid sequence metadata')
        digest=hashlib.sha256(json.dumps(r['frames'],sort_keys=True).encode()).hexdigest()
        for key,seen in ((r['group'],groups),(digest,digests)):
            if key in seen and seen[key]!=r['split']:raise ValueError('Group or duplicate sequence crosses partitions')
            seen[key]=r['split']
    subsets={}
    for split in ('train','validation','test'):
        part=[r for r in rows if r['split']==split]
        if set(r['label'] for r in part)!=set(MOTION_LABELS):raise ValueError('Each split needs J, Z and NONE background sequences')
        subsets[split]=(np.array([motion_features(r['frames']) for r in part]),np.array([r['label'] for r in part]))
    candidates={};scores={}
    for c in (1,10,100):
        model=make_pipeline(StandardScaler(),SVC(C=c,probability=True,random_state=42))
        model.fit(*subsets['train']);candidates[c]=model
        scores[c]=float(f1_score(subsets['validation'][1],model.predict(subsets['validation'][0]),average='macro'))
    best=max(scores,key=scores.get);model=candidates[best]
    report={'selection':'validation macro F1 only','validation_macro_f1':scores,'selected_C':best,'groups':groups,'counts':{s:len(v[1]) for s,v in subsets.items()},'test':classification_report(subsets['test'][1],model.predict(subsets['test'][0]),output_dict=True,zero_division=0),'note':'Clip classification only, not continuous translation. Group identifiers must reflect verified signer/session identities. Scores are heuristic; background coverage is limited.'}
    joblib.dump({'model':model,'version':MOTION_VERSION,'threshold':.7},'models/motion.joblib')
    Path('data/processed/motion-evaluation.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':print(json.dumps(train(),indent=2))
