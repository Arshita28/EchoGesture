"""Validation-only upgrade selection. Original test scores are not recomputed."""
import json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from asl_lab.geometry import geometry_features
from asl_lab.train import load_records
from asl_lab.features import FEATURE_VERSION

rows=load_records('data/processed/landmarks.jsonl')
train=[r for r in rows if r['split']=='train']; val=[r for r in rows if r['split']=='validation']
x=np.array([r['features'] for r in train]); y=np.array([r['label'] for r in train]); xv=np.array([r['features'] for r in val]); yv=np.array([r['label'] for r in val])
labels=sorted(set(y)); baseline=joblib.load('models/classifier.joblib'); results={}; fitted={}
models={'baseline':baseline['model']}
for c in [1,10,100]:
    for gamma in ['scale',.001,.01]:
        name=f'geometry_C{c}_gamma{gamma}'
        models[name]=make_pipeline(FunctionTransformer(geometry_features),StandardScaler(),SVC(C=c,gamma=gamma,probability=True,random_state=42))
for name,model in models.items():
    if name!='baseline': model.fit(x,y)
    pred=model.predict(xv)
    results[name]={'validation_macro_f1':float(f1_score(yv,pred,average='macro')),'classification_report':classification_report(yv,pred,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(yv,pred,labels=labels).tolist()}
    fitted[name]=model
    print(name,results[name]['validation_macro_f1'],flush=True)
winner=max(results,key=lambda k:results[k]['validation_macro_f1'])
report={'selection':'validation only; test signer already inspected previously, fresh independent evaluation still required','labels':labels,'selected':winner,'candidates':results,'training_groups':sorted(set(r['group'] for r in train)),'validation_groups':sorted(set(r['group'] for r in val))}
Path('reports/upgrades/geometry-validation.json').write_text(json.dumps(report,indent=2))
joblib.dump({**baseline,'model':fitted[winner],'name':winner,'feature_version':FEATURE_VERSION},'models/geometry-candidate.joblib')
print('Saved candidate separately; live baseline unchanged:',winner)
