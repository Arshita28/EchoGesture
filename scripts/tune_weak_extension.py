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
v=np.stack([summarize(x) for x in d['val_x']]);truth=np.array([labels[int(i)] for i in d['val_y']])
targets={'paper','study','tell','graduate','thursday'}
mask=np.array([w in targets for w in truth])
baseline=base.predict(v)
best=None
for c in (1,10,100,1000):
 for gamma in ('scale',.0001,.0005,.001):
  model=clone(base).set_params(C=c,gamma=gamma).fit(x,y)
  pred=model.predict(v)
  total=int(sum(pred==truth));weak=int(sum(pred[mask]==truth[mask]))
  eligible=total>=int(sum(baseline==truth))-2
  rank=(eligible,weak,total)
  if best is None or rank>best[0]:best=(rank,model)
_,candidate=best
report=dict(added=dict(Counter(w for w,_ in extra)),validation_baseline=int(sum(baseline==truth)),validation_candidate=int(sum(candidate.predict(v)==truth)),validation_total=len(truth),weak_baseline=int(sum(baseline[mask]==truth[mask])),weak_candidate=int(sum(candidate.predict(v)[mask]==truth[mask])),weak_total=int(mask.sum()),C=candidate.C,gamma=candidate.gamma)
joblib.dump(candidate,'models/word80-weak-tuned-candidate.joblib')
print(report);Path('reports/word80-weak-tuned.json').write_text(json.dumps(report,indent=2))
