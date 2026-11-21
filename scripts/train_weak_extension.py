import json,hashlib,joblib,numpy as np
from pathlib import Path
from sklearn.base import clone
from asl_lab.word_dense import extract_dense
from asl_lab.word_features import WordExtractor
from asl_lab.word_svm_features import summarize
from scripts.extract_words import quality
d=np.load('/tmp/echo80-neural-data.npz');labels=d['labels'].tolist()
base=joblib.load('models/word80-priority-wlasl-citizen-svm.joblib')
rows=json.load(open('datasets/raw/word-weak-extension/acquisition.json'))
e=WordExtractor();extra=[]
try:
 for r in rows:
  if r['status']!='acquired':continue
  try:
   x=extract_dense(r['path'],e)[0]
   if quality(x)['quality_pass']:extra.append((r['label'],summarize(x)))
  except ValueError:continue
finally:e.close()
from collections import Counter
x=np.stack([summarize(v) for v in d['train_x']]+[v for _,v in extra]);y=[labels[int(v)] for v in d['train_y']]+[w for w,_ in extra]
candidate=clone(base).fit(x,y)
v=np.stack([summarize(x) for x in d['val_x']]);truth=np.array([labels[int(i)] for i in d['val_y']])
report=dict(added=dict(Counter(w for w,_ in extra)),validation_baseline=int(sum(base.predict(v)==truth)),validation_candidate=int(sum(candidate.predict(v)==truth)),validation_total=len(truth))
joblib.dump(candidate,'models/word80-weak-candidate.joblib')
print(report);Path('reports/word80-weak-training.json').write_text(json.dumps(report,indent=2))
