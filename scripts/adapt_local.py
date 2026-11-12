"""Consented local personalization; never report same-session holdouts as independent accuracy."""
import csv,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,joblib
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from asl_lab.features import HandExtractor,FEATURE_VERSION
from asl_lab.geometry import geometry_features
from asl_lab.train import load_records

rows=load_records('data/processed/landmarks.jsonl');tr=[r for r in rows if r['split']=='train'];va=[r for r in rows if r['split']=='validation'];x=np.array([r['features'] for r in tr]);y=np.array([r['label'] for r in tr]);xv=np.array([r['features'] for r in va]);yv=np.array([r['label'] for r in va]);local=[];seen=set();h=HandExtractor()
try:
 for r in csv.DictReader(open('data/manifest.csv')):
  if not ((r['group']=='ms_debug' and r['label'] in 'AMS') or r['group']=={'D':'for_d','P':'for_P','E':'for_E','Q':'for_Q','U':'for_U','X':'for_X','G':'for_G','C':'for_C','I':'for_I'}.get(r['label'])):continue
  digest=hashlib.sha256((Path('data')/r['path']).read_bytes()).hexdigest()
  if digest in seen:continue
  seen.add(digest)
  f,status=h.extract(np.array(Image.open(Path('data')/r['path']).convert('RGB')))
  if f is not None:local.append({**r,'features':f.tolist()})
finally:h.close()
counts={};training=[];diagnostic=[]
totals={label:sum(r['label']==label for r in local) for label in set(r['label'] for r in local)}
for r in local:
 n=counts.get(r['label'],0);counts[r['label']]=n+1
 (training if n<min(3,totals[r['label']]-1) else diagnostic).append(r)
if set(r['label'] for r in training)!=set('AMSDPEQUXGCI'):raise ValueError('Need detected examples for all adaptation letters')
xa=np.vstack([x,np.array([r['features'] for r in training])]);ya=np.concatenate([y,[r['label'] for r in training]])
base=joblib.load('models/classifier.joblib');baseline=float(f1_score(yv,base['model'].predict(xv),average='macro'));results=[];fitted=[]
for weight in [1,3,10]:
 m=make_pipeline(FunctionTransformer(geometry_features),StandardScaler(),SVC(C=100,gamma='scale',probability=True,random_state=42));m.fit(xa,ya,svc__sample_weight=np.concatenate([np.ones(len(x)),np.full(len(training),weight)]));score=float(f1_score(yv,m.predict(xv),average='macro'))
 d=[]
 for r in diagnostic:
  p=m.predict_proba(np.array(r['features']).reshape(1,-1))[0];i=int(np.argmax(p));d.append({'expected':r['label'],'candidate':str(m.classes_[i]),'score':float(p[i])})
 results.append({'weight':weight,'public_validation_macro_f1':score,'same_session_diagnostics':d});fitted.append(m);print(weight,score,d,flush=True)
best=max(range(len(results)),key=lambda i:results[i]['public_validation_macro_f1'])
report={'note':'Up to three detected frames per letter (always reserving at least one) explicitly reassigned to adaptation training with user consent. Remaining frames are correlated same-session diagnostics, NOT independent evaluation. Public test report untouched. Fresh recordings required.','baseline_public_validation_macro_f1':baseline,'selected_weight':results[best]['weight'],'adaptation_train':[{'path':r['path'],'label':r['label']} for r in training],'reserved_diagnostics':[{'path':r['path'],'label':r['label']} for r in diagnostic],'results':results}
Path('data/processed/local-adaptation.json').write_text(json.dumps(report,indent=2));joblib.dump({**base,'model':fitted[best],'name':'geometry_local_adaptation','feature_version':FEATURE_VERSION},'models/local-adaptation-candidate.joblib')
print('Saved separate candidate; no live changes yet.')
