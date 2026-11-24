"""Compare complete browser-captured gestures, including trials with no capture."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from asl_lab.motion import motion_features
from asl_lab.motion_path import path_evidence

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--candidate',default='models/motion-extension-z-specialist-candidate.joblib')
parser.add_argument('--reference')
args=parser.parse_args()
models={name:joblib.load(path) for name,path in {
    'before':args.reference or ('models/motion-before-jz-extension.joblib' if Path('models/motion-before-jz-extension.joblib').exists() else 'models/motion.joblib'),
    'candidate':args.candidate}.items()}
rows=json.loads(Path('datasets/preprocessed/motion-extension/capture-replay.json').read_text())
results=[]
for row in rows:
    result={k:v for k,v in row.items() if k!='clips'}
    for name,bundle in models.items():
        decisions=[]
        for frames in row['clips']:
            try:
                x=motion_features(frames).reshape(1,-1)
            except ValueError:
                decisions.append(dict(label=None,top=None,error='invalid_capture'))
                continue
            scores=bundle['model'].predict_proba(x)[0]
            i=int(np.argmax(scores));label=str(bundle['model'].classes_[i]);score=float(scores[i])
            evidence=path_evidence(frames)
            accepted=label if score>=bundle['threshold'] and label in evidence.get('eligible',[]) else None
            accepted=evidence['label'] or accepted
            decisions.append(dict(label=accepted,top=label,score=score))
        expected=None if row['label']=='NONE' else row['label']
        result[name]=dict(decisions=decisions,correct=(not decisions and expected is None) or (bool(decisions) and decisions[-1]['label']==expected))
    results.append(result)
report=dict(rows=results,summary={})
for name in models:
    report['summary'][name]={}
    for label in ('J','Z','NONE'):
        subset=[r for r in results if r['label']==label]
        report['summary'][name][label]=[sum(r[name]['correct'] for r in subset),len(subset)]
Path('reports/motion-capture-replay.json').write_text(json.dumps(report,indent=2))
print(report['summary'])
