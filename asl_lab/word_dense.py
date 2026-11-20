"""Dense tracking and visible-sign cropping before fixed-length word sampling."""
import cv2
import numpy as np
from .word_features import WORD_FEATURE_DIM
from scripts.extract_words import selected_indices

DENSE_VERSION='dense-tracking-visible-crop-v1'


def crop_sequence(features,timestamps):
    features=np.asarray(features,dtype=np.float32);timestamps=np.asarray(timestamps,dtype=float)
    if features.ndim!=2 or features.shape[1]!=WORD_FEATURE_DIM or len(features)!=len(timestamps):
        raise ValueError('Invalid dense sequence dimensions')
    if not np.isfinite(features).all() or not np.isfinite(timestamps).all() or (np.diff(timestamps)<=0).any():
        raise ValueError('Invalid dense sequence values or timestamps')
    visible=(features[:,-1]>.5)&(features[:,-3:-1].max(axis=1)>.5)
    found=np.flatnonzero(visible)
    if len(found)<8:raise ValueError('Fewer than eight hand/body-visible observations')
    # Do not invent hand coordinates; retain internal gaps. Outer idle frames
    # without a detected hand do not dominate the resampled sign window.
    start,end=int(found[0]),int(found[-1])+1
    index=selected_indices(end-start)+start
    return features[index],timestamps[index],dict(start_frame=start,end_frame_exclusive=end,
        dense_frames=len(features),visible_frames=int(visible.sum()),cropped_duration_s=float(timestamps[end-1]-timestamps[start]))


def extract_dense(path,extractor,max_duration=10):
    cap=cv2.VideoCapture(str(path));features=[];times=[]
    try:
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        if not cap.isOpened() or not 0<fps<=120:raise ValueError('Invalid video')
        extractor.model.reset();index=0
        # Track chronologically at up to 25 FPS, instead of large jumps between 32 frames.
        next_time=0.
        while True:
            ok,frame=cap.read()
            if not ok:break
            timestamp=index/fps
            if timestamp>max_duration:raise ValueError('Video longer than allowed isolated clip')
            if timestamp+1e-7>=next_time:
                h,w=frame.shape[:2]
                if max(h,w)>640:frame=cv2.resize(frame,(round(w*640/max(h,w)),round(h*640/max(h,w))))
                features.append(extractor.extract(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
                times.append(timestamp);next_time+=1/25
            index+=1
        return crop_sequence(features,times)
    finally:cap.release()
