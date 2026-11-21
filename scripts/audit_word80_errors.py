import csv,json,joblib,numpy as np
from pathlib import Path
from collections import defaultdict
from asl_lab.word_svm_features import summarize
from scripts.extract_words import quality
from scripts.train_word_background import load
model=joblib.load('models/word80-priority-wlasl-citizen-svm.joblib')
root=Path('datasets/raw/asl-citizen-selected')
out={str(w):dict(attempted=0,correct=0,detection_failures=0,confusions={}) for w in model.classes_}
for r in csv.DictReader((root/'test.csv').open()):
 w=r['Gloss'].lower()
 if w not in out:continue
 a=out[w];a['attempted']+=1;p=(root/r['Video file']).with_suffix('.npy')
 if not p.exists() or not quality(np.load(p))['quality_pass']:a['detection_failures']+=1;continue
 pred=str(model.predict([summarize(np.load(p))])[0]);a['correct']+=int(pred==w)
 if pred!=w:a['confusions'][pred]=a['confusions'].get(pred,0)+1
for w,a in out.items():a['accuracy']=a['correct']/a['attempted'] if a['attempted'] else None
rows=[]
for folder in ['word20-dense','word-negative-dense']:
 rows+=json.load(open('datasets/preprocessed/'+folder+'/extraction.json'))['rows']
wl=[]
for r in rows:
 if r['label'] in out and r['split']=='test' and r['quality_pass']:
  pred=str(model.predict([summarize(load(r))])[0]);wl.append(dict(word=r['label'],prediction=pred))
report=dict(words=out,wlasl_test=wl)
Path('reports/word80-full-error-audit.json').write_text(json.dumps(report,indent=2))
ranked=sorted([(w,a) for w,a in out.items() if a['attempted']],key=lambda p:p[1]['accuracy'])
print('covered',len(ranked),'of',len(out),'correct',sum(a['correct'] for a in out.values()),'total',sum(a['attempted'] for a in out.values()))
print('weakest',[(w,a['correct'],a['attempted']) for w,a in ranked[:15]])
print('no coverage',[w for w,a in out.items() if not a['attempted']])
print('wlasl',sum(r['word']==r['prediction'] for r in wl),len(wl))
