import csv,json,struct,zlib,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np,joblib
from asl_lab.word_dense import extract_dense
from asl_lab.word_features import WordExtractor
from scripts.extract_words import quality
from scripts.train_word_background import load
from asl_lab.word_model import WordTemplateClassifier
from collections import Counter
root=Path('datasets/raw/asl-citizen-selected')
entries=json.loads((root/'entries.json').read_text())
labels=json.load(open('config/word20.json'))['labels']
url='https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip'
selected=[]
for split in ['train','val']:
 for r in csv.DictReader((root/(split+'.csv')).open()):
  if r['Gloss'].lower() in labels:selected.append((split,r))
def acquire(pair):
 split,r=pair
 name=r['Video file'];p=root/name
 if not p.exists():
  start,size,length,crc,method=entries[name]
  b=subprocess.check_output(['curl','-fsSL','--retry','2','--max-time','60','--range',f'{start}-{start+size+2047}',url+'?offset='+str(start)])
  assert b[:4]==b'PK\x03\x04'
  a,c=struct.unpack_from('<HH',b,26);raw=b[30+a+c:30+a+c+size]
  data=zlib.decompress(raw,-15) if method==8 else raw
  assert len(data)==length and zlib.crc32(data)==crc
  p.write_bytes(data)
 return p
with ThreadPoolExecutor(3) as pool:paths=list(pool.map(acquire,selected))
print('downloaded',len(paths),flush=True)
tracker=WordExtractor(); groups={'train':[],'val':[]}
try:
 for p,(split,r) in zip(paths,selected):
  cache=p.with_suffix('.npy')
  try:
   if not cache.exists():np.save(cache,extract_dense(p,tracker)[0])
   x=np.load(cache)
   if quality(x)['quality_pass']:groups[split].append((r['Gloss'].lower(),x.reshape(-1)))
  except ValueError:continue
finally:tracker.close()
rows=json.load(open('datasets/preprocessed/word20-dense/extraction.json'))['rows']
train=[(r['label'],load(r)) for r in rows if r['quality_pass'] and r['split']=='train']+groups['train']
base=joblib.load('models/words20-candidate.joblib')
model=WordTemplateClassifier().fit([x for _,x in train],[y for y,_ in train])
report={'added':dict(Counter(y for y,_ in groups['train'])),'validation':{}}
for src,val in [('citizen',groups['val']),('wlasl',[(r['label'],load(r)) for r in rows if r['quality_pass'] and r['split']=='validation'])]:
 y=[a for a,_ in val];x=[b for _,b in val]
 report['validation'][src]={k:int(np.sum(m.predict(x)==np.array(y))) for k,m in [('baseline',base['model']),('expanded',model)]}
 report['validation'][src]['total']=len(y)
joblib.dump(dict(base,model=model,name='word20_citizen_candidate'),'models/word20-citizen.joblib')
Path('reports/word20-citizen-training.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
