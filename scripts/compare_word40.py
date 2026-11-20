import csv,json,re,struct,zlib,subprocess,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import numpy as np,joblib
from sklearn.base import clone
from scripts.compare_citizen_svm import word_dataset
from asl_lab.word_svm_features import summarize
from asl_lab.word_features import WordExtractor
from asl_lab.word_dense import extract_dense
from scripts.extract_words import quality
root=Path('datasets/raw/asl-citizen-selected')
old=json.load(open('config/word20.json'))['labels']
common={e['gloss'] for e in json.load(open('datasets/raw/wlasl/WLASL_v0.3.json'))}
splits={s:list(csv.DictReader((root/(s+'.csv')).open())) for s in ('train','val','test')}
def matches(w,s):return [r for r in splits[s] if r['Gloss']==w.upper()]
pool={r['Gloss'].lower() for r in splits['train']}&common-set(old)
counts={s:Counter(r['Gloss'].lower() for r in rows) for s,rows in splits.items()}
pool=[w for w in pool if w.isalpha() and counts['val'][w]>=4 and counts['test'][w]>=10]
new=sorted(pool,key=lambda w:(-len({r['Participant ID'] for r in matches(w,'train')}),-counts['train'][w],w))[:20]
print(new,flush=True)
entries=json.load(open(root/'entries.json'))
url='https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip'
groups={}
tracker=WordExtractor()
def get(r):
 name=r['Video file'];p=root/name
 if not p.exists():
  start,size,length,crc,method=entries[name]
  for attempt in range(4):
   try:
    b=subprocess.check_output(['curl','-fsSL','--max-time','45','--range',f'{start}-{start+size+2047}',url+f'?offset={start}&try={attempt}'])
    a,c=struct.unpack_from('<HH',b,26);raw=b[30+a+c:30+a+c+size];data=zlib.decompress(raw,-15) if method==8 else raw
    assert len(data)==length and zlib.crc32(data)==crc
    p.write_bytes(data);break
   except Exception:
    if attempt==3:raise
 return p
for s in splits:
 rows=[r for r in splits[s] if r['Gloss'].lower() in new]
 with ThreadPoolExecutor(8) as pool:paths=list(pool.map(get,rows))
 groups[s]=[]
 for r,p in zip(rows,paths):
  cache=p.with_suffix('.npy')
  try:
   if not cache.exists():np.save(cache,extract_dense(p,tracker)[0])
   x=np.load(cache)
   if quality(x)['quality_pass']:groups[s].append((r['Gloss'].lower(),summarize(x)))
  except ValueError:continue
 print(s,len(groups[s]),'of',len(rows),flush=True)
tracker.close()
base=joblib.load('models/word20-citizen-svm.joblib')
y,x,_=word_dataset('train')
candidate=clone(base).fit(np.stack([summarize(a) for a in x]+[a for _,a in groups['train']]),list(y)+[w for w,_ in groups['train']])
report={'added':new,'new_train_counts':dict(Counter(w for w,_ in groups['train']))}
for s,role in [('val','validation'),('test','test')]:
 y,x,src=word_dataset(role);v=np.stack([summarize(a) for a in x])
 report[s]={}
 for source in set(src):
  idx=src==source;report[s][str(source)]=[int(sum(m.predict(v[idx])==y[idx])) for m in (base,candidate)]+[int(idx.sum())]
 ys,vs=zip(*groups[s]);report[s]['added']=[int(sum(candidate.predict(vs)==np.array(ys))),sum(counts[s][w] for w in new)]
joblib.dump(candidate,'models/word40-citizen-svm.joblib')
Path('reports/word40-comparison.json').write_text(json.dumps(report,indent=2))
print(report,flush=True)
