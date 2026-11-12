"""Fixed-parameter local J/Z/background adaptation; reserved clips are development checks."""
import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from asl_lab.motion import motion_features
ROOT=Path(__file__).resolve().parents[1]
public=json.loads((ROOT/'data/processed/motion-public/sequences.json').read_text())
clips=[{**json.loads(p.read_text()),'file':p.name} for p in sorted((ROOT/'data/processed/motion').glob('*.json'),key=lambda p:p.stat().st_mtime)]
by={label:[r for r in clips if r['label']==label] for label in ['J','Z','NONE']}
if len(by['J'])<4 or len(by['Z'])<5 or len(by['NONE'])<5:raise ValueError('Need more labeled clips')
# Preserve the original first Z clip as a diagnostic; retain approved two J roles.
training=by['J'][:2]+by['Z'][1:4]+by['NONE'][:3]
training_names={r['file'] for r in training};reserved=[r for r in clips if r['file'] not in training_names]
hashes={hashlib.sha256(json.dumps(r['frames'],sort_keys=True).encode()).hexdigest() for r in training}
for r in reserved:
 if hashlib.sha256(json.dumps(r['frames'],sort_keys=True).encode()).hexdigest() in hashes:raise ValueError('Duplicate clip crosses roles')
x=np.array([motion_features(r['frames']) for r in public+training]);y=np.array([r['label'] for r in public+training])
model=make_pipeline(StandardScaler(),SVC(C=1,class_weight='balanced',probability=True,random_state=42))
model.fit(x,y,svc__sample_weight=np.concatenate([np.ones(len(public)),np.full(len(training),3.0)]))
base=joblib.load(ROOT/'models/motion.joblib');results=[]
for r in reserved:
 f=motion_features(r['frames']).reshape(1,-1);o={'file':r['file'],'expected':r['label']}
 for name,m in [('before',base['model']),('candidate',model)]:
  scores=m.predict_proba(f)[0];i=int(np.argmax(scores));label=str(m.classes_[i]);confidence=float(scores[i]);o[name]={'candidate':label,'score':confidence,'accepted':label if label!='NONE' and confidence>=.7 else None}
 results.append(o)
summary={}
for name in ['before','candidate']:
 summary[name]={label:{'clips':sum(r['expected']==label for r in results),'correct_decisions':sum(r['expected']==label and r[name]['accepted']==(None if label=='NONE' else label) for r in results)} for label in ['J','Z','NONE']}
bundle={**base,'model':model,'validation_status':'local_balanced_adaptation_not_independently_validated','training_count':len(public)+len(training)}
joblib.dump(bundle,ROOT/'models/motion-balanced-candidate.joblib')
report={'note':'Fixed C=1/weight=3 before fitting. Original two approved J clips plus selected new Z/background clips are adaptation training. Other user clips excluded from fitting but same-user, already-inspected development comparisons; NOT independent evaluation. Static/voice features untouched.','training_files':[r['file'] for r in training],'summary':summary,'reserved_results':results}
(ROOT/'data/processed/motion-balanced-adaptation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
