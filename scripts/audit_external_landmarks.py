import cv2,json,numpy as np
from pathlib import Path
from asl_lab.word_features import WordExtractor
root=Path('mytest');out=[]
for w in ['water','apple','red','green']:
 cap=cv2.VideoCapture(str(root/(w+'.mp4')));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));targets=set(np.linspace(0,n-1,6).astype(int));e=WordExtractor();tiles=[]
 try:
  for i in range(n):
   ok,img=cap.read()
   if not ok:break
   h,b=img.shape[:2];res=e.model.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
   if i not in targets:continue
   for hand in [res.left_hand_landmarks,res.right_hand_landmarks]:
    if hand:
     for a in hand.landmark:cv2.circle(img,(int(a.x*b),int(a.y*h)),3,(0,255,150),-1)
   tile=cv2.resize(img,(240,180));cv2.putText(tile,f'{w} frame {i}',(7,170),0,.4,(255,255,255),1);tiles.append(tile)
 finally:cap.release();e.close()
 out.append(np.hstack(tiles))
cv2.imwrite('mytest/tracking-audit.jpg',np.vstack(out))
