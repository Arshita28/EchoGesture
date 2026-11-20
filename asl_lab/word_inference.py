"""Local video -> experimental known-word prediction. No capture is retained."""
from pathlib import Path
import time
import joblib
from .word_features import WordExtractor
from .word_model import classify_word
from scripts.acquire_wlasl import inspect_video
from scripts.extract_words import extract_clip
from .word_dense import extract_dense,DENSE_VERSION


def predict(video_path, model_path=None, bundle=None):
    started=time.perf_counter()
    if bundle is None:
        model_path=Path(model_path) if model_path else Path(__file__).resolve().parents[1]/'models/words-prototype.joblib'
        bundle=joblib.load(model_path)  # Trusted local artifact only.
    info=inspect_video(video_path,max_frames=1200)
    if info['duration_s']>10:raise ValueError('Use one isolated sign in a video of at most 10 seconds')
    extractor=WordExtractor()
    try:
        preprocessing=bundle.get('preprocessing')
        if preprocessing==DENSE_VERSION:features,timestamps,extraction=extract_dense(video_path,extractor)
        elif preprocessing is None:features,timestamps,extraction=extract_clip(video_path,extractor)
        else:raise ValueError('Unknown word preprocessing version')
    finally:extractor.close()
    result=classify_word(bundle,features)
    result.update(model=bundle['name'],supported_words=bundle['labels'],
                  processing_ms=round((time.perf_counter()-started)*1000,1),
                  clip_duration_s=round(info['duration_s'],2),preprocessing=bundle.get('preprocessing','sparse-32-v1'))
    return result
