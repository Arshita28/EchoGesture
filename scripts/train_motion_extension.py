"""Train J/Z templates with extra dictionary clips; preserve fixed evaluation roles."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

import joblib
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from asl_lab.motion_summary import MotionSummary, MotionZSpecialist
from asl_lab.motion import motion_features
from asl_lab.temporal_match import TemporalMatcher, MotionTrajectoryMatcher
from asl_lab.motion_path import path_evidence

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--method',choices=['templates','trajectory','svm','z-specialist'],default='templates')
parser.add_argument('--public-motion',action='store_true')
parser.add_argument('--i-safe',action='store_true')
args=parser.parse_args()
if args.i_safe and (not args.public_motion or args.method!='svm'):
    parser.error('--i-safe requires --public-motion --method svm')
stem='i-safe' if args.i_safe else ('public-expanded' if args.public_motion else 'extension')
root = Path('datasets/preprocessed')
reference=Path('models/motion-before-jz-extension.joblib')
base=joblib.load('models/motion.joblib' if args.public_motion else (reference if reference.exists() else 'models/motion.joblib'))
public = json.loads((root / 'motion-public/sequences.json').read_text())
roles = json.loads((root / 'motion-balanced-adaptation.json').read_text())
local = [json.loads((root / 'motion' / name).read_text()) for name in roles['training_files']]
extension = json.loads((root / 'motion-extension/sequences.json').read_text())
train = public + local + [r for r in extension if r['role']=='train' and r['usable']]
if args.public_motion:
    train += json.loads((root/'public-jz-landmarks/sequences.json').read_text())
excluded_j=[]
synthetic_i=[]
if args.i_safe:
    excluded_j=[r for r in train if r['label']=='J' and path_evidence(r['frames'])['label']!='J']
    train=[r for r in train if r['label']!='J' or path_evidence(r['frames'])['label']=='J']
    synthetic_i=[r for r in json.loads((root/'i-motion-negatives/sequences.json').read_text()) if r['role']=='train']
    train+=synthetic_i
held = [dict(**json.loads((root / 'motion' / r['file']).read_text()),source='local_development') for r in roles['reserved_results']]
held += [{**r,'source':'dictionary_reserved'} for r in extension if r['role']=='test' and r['usable']]
def signature(frames):
    return hashlib.sha256(json.dumps(frames,sort_keys=True).encode()).hexdigest()
if {signature(r['frames']) for r in train} & {signature(r['frames']) for r in held}:
    raise ValueError('Duplicate sequence across training/evaluation')

x, y, weights = [], [], []
capture_path=root/'motion-extension'/('capture-public-training.json' if args.public_motion else 'capture-training.json')
derived=[]
if capture_path.exists():
    for row in json.loads(capture_path.read_text()):
        for frames in row['clips']:
            try:motion_features(frames)
            except ValueError:continue
            derived.append(dict(label=row['label'],frames=frames,source=row.get('source')))
if args.i_safe:
    derived=[r for r in derived if r['label']!='J' or path_evidence(r['frames'])['label']=='J']
for row in train+derived:
    original = motion_features(row['frames']).reshape(24,21,3)
    for mirror in (False,True):
        for amplitude in (.75,1.,1.25):
            seq=original.copy()
            # Change travel size while retaining each frame's local handshape.
            shift=seq[:,0]-seq[0,0]
            seq+=(amplitude-1)*shift[:,None]
            if mirror:seq[:,:,0]*=-1
            x.append(seq.reshape(-1));y.append(row['label']);weights.append(.1 if row.get('source')=='public_jz_single_signer' else (.25 if row.get('source')=='synthetic_i_stress' else 1.))
if args.method in ('svm','z-specialist'):
    candidate=make_pipeline(MotionSummary(),StandardScaler(),SVC(C=10,probability=True,class_weight='balanced',random_state=42)).fit(x,y,svc__sample_weight=np.asarray(weights))
else:
    matcher=TemporalMatcher if args.method=='templates' else MotionTrajectoryMatcher
    candidate=matcher().fit(x,y)
if args.method=='z-specialist':
    candidate=MotionZSpecialist(base['model'],candidate)
report=dict(excluded_incomplete_j=len(excluded_j),synthetic_i_sequences=len(synthetic_i),public_motion=args.public_motion, public_source_weight=.1 if args.public_motion else None, training_clips=dict(Counter(r['label'] for r in train)),templates=len(x),derived_capture_sequences=len(derived),rows=[],
            limitations='Dictionary roles fixed before prediction; external signer overlap unverified. Local checks previously inspected. Augmentations are derived training examples.')
for row in held:
    feature=motion_features(row['frames']).reshape(1,-1)
    result=dict(expected=row['label'],source=row['source'])
    evidence=path_evidence(row['frames'])
    for name,model in [('before',base['model']),('candidate',candidate)]:
        scores=model.predict_proba(feature)[0];i=int(np.argmax(scores))
        label=str(model.classes_[i]);score=float(scores[i])
        accepted=label if label!='NONE' and score>=.7 and label in evidence.get('eligible',[]) else None
        if evidence['label']:accepted=evidence['label']
        result[name]=dict(top=label,score=score,accepted=accepted,correct=accepted==(None if row['label']=='NONE' else row['label']))
    report['rows'].append(result)
report['summary']={}
for source in ('local_development','dictionary_reserved'):
    report['summary'][source]={}
    for name in ('before','candidate'):
        report['summary'][source][name]={}
        for label in ('J','Z','NONE'):
            rows=[r for r in report['rows'] if r['source']==source and r['expected']==label]
            report['summary'][source][name][label]=[sum(r[name]['correct'] for r in rows),len(rows)]
joblib.dump({**base,'model':candidate,'training_count':len(train),'augmentation_templates':len(x),
             'validation_status':'dictionary_and_inspected_local_development',
             'source':base['source']+'; dictionary sources in datasets/raw/motion-extension/sources.json'+('; JordiEncaboBataller/sign-language-cv raw single-person motion release' if args.public_motion else '')},
            f'models/motion-{stem}-{args.method}-candidate.joblib')
Path(f'reports/motion-{stem}-{args.method}.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report['summary'],indent=2),flush=True)
