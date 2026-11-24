import csv,json,joblib,numpy as np
from pathlib import Path
from sklearn.svm import SVC
from asl_lab.word_invariant import mirrored
from scripts.compare_citizen_svm import word_dataset
from scripts.train_word_background import load
from scripts.extract_words import quality
from asl_lab.word_dense import extract_dense
from asl_lab.word_features import WordExtractor
root=Path('datasets/raw/asl-citizen-selected');base=joblib.load('models/word80-priority-wlasl-citizen-svm.joblib');labels=set(base.classes_)
groups={}
for role,name in [('train','train'),('validation','val'),('test','test')]:
 y,x,_=word_dataset(role);data=list(zip(y,x))
 for r in csv.DictReader((root/(name+'.csv')).open()):
  if r['Gloss'].lower() not in labels-set(y):continue
  p=(root/r['Video file']).with_suffix('.npy')
  if p.exists():
   x=np.load(p)
   if quality(x)['quality_pass']:data.append((r['Gloss'].lower(),x.ravel()))
 wl=json.load(open('datasets/preprocessed/word-negative-dense/extraction.json'))['rows']
 data.extend((r['label'],load(r)) for r in wl if r['quality_pass'] and r['split']==role and r['label'] in labels-set(y))
 groups[role]=data
def features(x,bins):
 x=x.reshape(32,154).copy()
 if x[:,151].sum()>x[:,152].sum():x=mirrored(x)
 # Retain each hand's local shape and body-relative position over finer intervals.
 return np.concatenate([b.mean(0) for b in np.array_split(x,bins)]+[x.std(0)])
y=np.array([a for a,b in groups['train']]);vy=np.array([a for a,b in groups['validation']])
best=None;trials=[]
for bins in [8,16]:
 x=np.stack([features(b,bins) for a,b in groups['train']]);v=np.stack([features(b,bins) for a,b in groups['validation']])
 for c in [10,100,1000]:
  m=SVC(C=c,class_weight='balanced').fit(x,y);score=int(sum(m.predict(v)==vy));trials.append(dict(bins=bins,C=c,correct=score,total=len(vy)))
  if best is None or score>best[0]:best=(score,bins,m)
_,bins,m=best;test=groups['test'];pred=m.predict(np.stack([features(b,bins) for a,b in test]));report=dict(trials=trials,bins=bins,C=m.C,test_correct=sum(int(a==p) for (a,b),p in zip(test,pred)),test_usable=len(test))
e=WordExtractor();rows=[]
try:
 for r in json.load(open('mytest/sources.json')):
  x=extract_dense(Path('mytest')/r['file'],e)[0];rows.append(dict(word=r['word'],prediction=str(m.predict([features(x,bins)])[0])))
finally:e.close()
report['regression']=rows
Path('reports/word80-temporal-candidate.json').write_text(json.dumps(report,indent=2))
joblib.dump(dict(model=m,bins=bins),'models/word80-temporal-candidate.joblib')
print(report)
