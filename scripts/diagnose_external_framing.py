import cv2,json,joblib,numpy as np
from pathlib import Path
from asl_lab.word_dense import extract_dense
from asl_lab.word_features import WordExtractor
from asl_lab.word_svm_features import summarize
class Cropped(WordExtractor):
 def extract(self,rgb):
  h,w=rgb.shape[:2]
  return super().extract(rgb[int(h*.05):int(h*.95),int(w*.2):int(w*.8)])
m=joblib.load('models/word80-priority-wlasl-citizen-svm.joblib')
e=Cropped();out=[]
try:
 for r in json.load(open('mytest/sources.json')):
  x=extract_dense(Path('mytest')/r['file'],e)[0]
  out.append(dict(word=r['word'],pred=str(m.predict([summarize(x)])[0])))
finally:e.close()
print(out)
Path('reports/mytest-framing-diagnostic.json').write_text(json.dumps(out,indent=2))
