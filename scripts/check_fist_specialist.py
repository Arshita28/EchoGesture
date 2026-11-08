"""Validation-only closed-fist experiment; diagnostic webcam photos never fitted."""
import sys,json,joblib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from asl_lab.geometry import geometry_features
from asl_lab.train import load_records
rows=load_records('data/processed/landmarks.jsonl');tr=[r for r in rows if r['split']=='train'];va=[r for r in rows if r['split']=='validation'];group=set('AEMNST');train=[r for r in tr if r['label'] in group]
x=np.array([r['features'] for r in train]);y=np.array([r['label'] for r in train]);xv=np.array([r['features'] for r in va]);yv=np.array([r['label'] for r in va]);base=joblib.load('models/classifier.joblib')['model'];pred=base.predict(xv);debug=[r for r in json.load(open('data/processed/ms-diagnostic.json')) if 'features' in r];xd=np.array([r['features'] for r in debug]);yd=np.array([r['label'] for r in debug]);out=[]
for c in [1,10,100]:
 model=make_pipeline(StandardScaler(),SVC(C=c,probability=True,random_state=42));model.fit(geometry_features(x)[:,63:],y);p=pred.copy();mask=np.array([v in group for v in p]);p[mask]=model.predict(geometry_features(xv[mask])[:,63:]);dp=model.predict(geometry_features(xd)[:,63:]);result={'C':c,'validation_macro_f1':float(f1_score(yv,p,average='macro')),'validation_M_recall':float((p[yv=='M']=='M').mean()),'diagnostic_predictions':list(zip(yd.tolist(),dp.tolist()))};out.append(result);print(result,flush=True)
Path('data/processed/fist-specialist-experiment.json').write_text(json.dumps({'note':'Not activated. Webcam photos excluded from fitting; diagnostic results are not independent test accuracy.','results':out},indent=2))
