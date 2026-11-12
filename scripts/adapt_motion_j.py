"""Consented two-clip J adaptation, with remaining user clips excluded from fitting."""
import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from asl_lab.motion import motion_features,MOTION_VERSION

ROOT=Path(__file__).resolve().parents[1]
public=json.loads((ROOT/'data/processed/motion-public/sequences.json').read_text())
paths=sorted((ROOT/'data/processed/motion').glob('*.json'),key=lambda p:p.stat().st_mtime)
clips=[{**json.loads(p.read_text()),'file':p.name} for p in paths]
j=[r for r in clips if r['label']=='J']
if len(j)<4:raise ValueError('Need at least four J clips for approved two-clip adaptation and reserved comparison')
training=j[:2];reserved=[r for r in clips if r not in training]
train_hashes={hashlib.sha256(json.dumps(r['frames'],sort_keys=True).encode()).hexdigest() for r in training}
for r in reserved:
 if hashlib.sha256(json.dumps(r['frames'],sort_keys=True).encode()).hexdigest() in train_hashes:raise ValueError('Duplicate sequence in reserved comparison')
x=np.array([motion_features(r['frames']) for r in public+training]);y=np.array([r['label'] for r in public+training])
model=make_pipeline(StandardScaler(),SVC(C=1,class_weight='balanced',probability=True,random_state=42))
model.fit(x,y,svc__sample_weight=np.concatenate([np.ones(len(public)),np.full(2,3.0)]))
base=joblib.load(ROOT/'models/motion.joblib');results=[]
for r in reserved:
 feature=motion_features(r['frames']).reshape(1,-1);row={'file':r['file'],'expected':r['label']}
 for name,m in [('before',base['model']),('candidate',model)]:
  scores=m.predict_proba(feature)[0];i=int(np.argmax(scores));label=str(m.classes_[i]);score=float(scores[i]);row[name]={'candidate':label,'score':score,'accepted':label if label!='NONE' and score>=.7 else None}
 results.append(row)
bundle={**base,'model':model,'version':MOTION_VERSION,'validation_status':'local_J_adaptation_not_independently_validated','training_count':len(public)+2}
joblib.dump(bundle,ROOT/'models/motion-J-candidate.joblib')
report={'note':'User explicitly approved two J clips for training. Other user clips excluded from fitting, but comparison is same-user and not verified session-independent. Mirror test did not resolve J. No static model/camera/voice changes. Fixed C=1, adaptation weight=3 before fitting. Fresh J/Z/background evaluation remains required.','training_files':[r['file'] for r in training],'reserved_results':results}
(ROOT/'data/processed/motion-J-adaptation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
