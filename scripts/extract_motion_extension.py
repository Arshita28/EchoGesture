"""Extract additional dictionary clips with fixed train/evaluation roles."""
import json
from pathlib import Path
import cv2
import numpy as np
from asl_lab.features import HandExtractor
from asl_lab.motion import motion_features

root = Path('datasets/raw/motion-extension')
rows = json.loads((root / 'sources.json').read_text())
cache=Path('datasets/preprocessed/motion-extension/sequences.json')
previous={r['sha256']:r for r in json.loads(cache.read_text())} if cache.exists() else {}
tracker = HandExtractor(static_image_mode=False)
try:
    for row in rows:
        if row['sha256'] in previous and previous[row['sha256']].get('usable') and previous[row['sha256']].get('extraction_version')=='moving-hand-v2':
            row.update(previous[row['sha256']])
            continue
        tracker.hands.reset()
        cap = cv2.VideoCapture(row['path'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames, index, next_ms = [], 0, 0.
        hands={'Left':[],'Right':[]}
        try:
            if not 0 < fps <= 120:
                raise ValueError('Invalid video rate')
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                ms = index*1000/fps
                if ms > 10000:
                    raise ValueError('Longer than 10 seconds, needs segmentation')
                if ms >= next_ms:
                    h,w=image.shape[:2]
                    if h<720:image=cv2.resize(image,(round(w*720/h),720))
                    found=tracker.hands.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                    for hand,side in zip(found.multi_hand_landmarks or [],found.multi_handedness or []):
                        points=[[p.x,p.y,p.z] for p in hand.landmark]
                        bucket=hands[side.classification[0].label]
                        if not bucket or bucket[-1]['time']!=ms:
                            bucket.append(dict(points=points,time=ms,aspect=image.shape[1]/image.shape[0]))
                    next_ms += 50
                index += 1
            def travel(frames):
                if len(frames)<6:return -1
                p=np.array([f['points'] for f in frames]);p[:,:,1]/=frames[0]['aspect']
                radius=np.median(np.linalg.norm(p-p[:,:1],axis=2).max(axis=1))
                spread=np.percentile(p[:,[0,8,20],:2],90,axis=0)-np.percentile(p[:,[0,8,20],:2],10,axis=0)
                return float(np.linalg.norm(spread,axis=1).max()/max(radius,1e-6))
            selected=max(hands,key=lambda side:travel(hands[side]))
            frames=hands[selected]
            row.update(selected_hand=selected,extraction_version='moving-hand-v2')
            runs=[[]]
            for frame in frames:
                if runs[-1] and frame['time']-runs[-1][-1]['time']>750:runs.append([])
                runs[-1].append(frame)
            frames=max(runs,key=len)
            if len(frames)>120:
                frames=[frames[i] for i in np.linspace(0,len(frames)-1,120).astype(int)]
            motion_features(frames)
            row.update(frames=frames, usable=True)
        except ValueError as error:
            row.update(usable=False,error=str(error))
        finally:
            cap.release()
        print(row['source_label'],row['role'],row['usable'],len(frames),flush=True)
finally:
    tracker.close()
Path('datasets/preprocessed/motion-extension').mkdir(exist_ok=True)
Path('datasets/preprocessed/motion-extension/sequences.json').write_text(json.dumps(rows))
