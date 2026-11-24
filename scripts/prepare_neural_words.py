import csv,json,numpy as np
from pathlib import Path
import joblib
from scripts.compare_citizen_svm import word_dataset
from scripts.extract_words import quality
from scripts.train_word_background import load
from asl_lab.word_invariant import mirrored
model=joblib.load('models/word80-priority-wlasl-citizen-svm.joblib');labels=list(model.classes_);root=Path('datasets/raw/asl-citizen-selected')
data={}
for role,n in [('train','train'),('validation','val'),('test','test')]:
 y,x,_=word_dataset(role);samples=list(zip(y,x));old=set(y)
 for r in csv.DictReader((root/(n+'.csv')).open()):
  w=r['Gloss'].lower();p=(root/r['Video file']).with_suffix('.npy')
  if w in set(labels)-old and p.exists() and quality(np.load(p))['quality_pass']:samples.append((w,np.load(p)))
 for r in json.load(open('datasets/preprocessed/word-negative-dense/extraction.json'))['rows']:
  if r['label'] in set(labels)-old and r['split']==role and r['quality_pass']:samples.append((r['label'],load(r)))
 xx=[]
 for w,s in samples:
  s=s.reshape(32,154)
  if s[:,151].sum()>s[:,152].sum():s=mirrored(s)
  xx.append(s)
 data[n+'_x']=np.array(xx);data[n+'_y']=np.array([labels.index(w) for w,s in samples])
np.savez_compressed('/tmp/echo80-neural-data.npz',labels=labels,**data)
