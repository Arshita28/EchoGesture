import cv2,json,mediapipe as mp
from pathlib import Path
out={}
for w in ['water','apple','red','green']:
 cap=cv2.VideoCapture('mytest/'+w+'.mp4');n=0;hol=0;hands=0;diff=[]
 with mp.solutions.holistic.Holistic(model_complexity=1,smooth_landmarks=False) as h, mp.solutions.hands.Hands(max_num_hands=2,min_detection_confidence=.3,min_tracking_confidence=.3) as d:
  while True:
   ok,img=cap.read()
   if not ok:break
   rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);a=h.process(rgb);b=d.process(rgb);one=bool(a.left_hand_landmarks or a.right_hand_landmarks);two=bool(b.multi_hand_landmarks)
   hol+=one;hands+=two
   if two and not one:diff.append(n)
   n+=1
 cap.release();out[w]=dict(frames=n,holistic=hol,independent_hands=hands,recovered_frames=diff)
print(out);Path('mytest/independent-tracking.json').write_text(json.dumps(out,indent=2))
