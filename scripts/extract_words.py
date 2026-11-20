"""Extract fixed-length word sequences from decoded, already isolated WLASL clips."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from asl_lab.word_features import WordExtractor, WORD_FEATURE_VERSION, WORD_FEATURE_DIM

SEQUENCE_FRAMES = 32
MIN_HAND_COVERAGE = 0.5
MIN_BODY_COVERAGE = 0.7


def selected_indices(total, length=SEQUENCE_FRAMES):
    if total < 8 or length < 8:
        raise ValueError('Need at least eight actual frames and sequence positions')
    return np.linspace(0, total-1, length).round().astype(int)


def extract_clip(path, extractor):
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not 0 < fps <= 120 or total > 4500:
            raise ValueError('Invalid video metadata or frame budget')
        indices = selected_indices(total)
        wanted = set(indices.tolist()); frames = {}; index = 0
        extractor.model.reset()  # No tracking state crosses clips or partitions.
        started = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted:
                h,w = frame.shape[:2]
                if max(h,w)>640:
                    frame=cv2.resize(frame,(round(w*640/max(h,w)),round(h*640/max(h,w))))
                frames[index] = extractor.extract(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index += 1
            if index > 4500:
                raise ValueError('Decoded video exceeds frame budget')
        if abs(index-total)>1 or any(i not in frames for i in wanted):
            raise ValueError('Video truncated or selected frames unavailable')
        features = np.stack([frames[i] for i in indices])
        if features.shape != (SEQUENCE_FRAMES, WORD_FEATURE_DIM) or not np.isfinite(features).all():
            raise ValueError('Invalid extracted feature shape or values')
        return features, indices / fps, dict(actual_decoded_frames=index,
            unique_sampled_frames=len(wanted), fps=float(fps), duration_s=index/fps,
            extraction_ms=round((time.perf_counter()-started)*1000,2))
    finally:
        cap.release()


def quality(features):
    masks = features[:, -3:]
    body = float(masks[:,2].mean()); hand = float((masks[:,:2].max(axis=1)>0).mean())
    return dict(body_coverage=body, any_hand_coverage=hand,
        both_hand_coverage=float((masks[:,:2].min(axis=1)>0).mean()),
        quality_pass=body >= MIN_BODY_COVERAGE and hand >= MIN_HAND_COVERAGE)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata',type=Path)
    parser.add_argument('--classes',type=int,default=30)
    parser.add_argument('--videos',type=Path,default=Path('datasets/raw/wlasl/videos'))
    parser.add_argument('--output',type=Path,default=Path('datasets/preprocessed/words'))
    parser.add_argument('--limit',type=int,default=None)
    args=parser.parse_args(); data=json.loads(args.metadata.read_text())
    if not 2<=args.classes<=len(data):parser.error('Invalid class count')
    args.output.mkdir(parents=True,exist_ok=True)
    entries=[]
    for label_id,entry in enumerate(data[:args.classes]):
        for item in entry['instances']:
            video_id=str(item['video_id'])
            if not video_id.isdigit():raise ValueError('Invalid video ID')
            path=args.videos/(video_id+'.mp4')
            if path.is_file():entries.append((label_id,entry['gloss'],item,path))
    if args.limit is not None:
        if args.limit<1:parser.error('Limit must be positive')
        entries=entries[:args.limit]
    extractor=WordExtractor(); rows=[]
    try:
        for label_id,label,item,path in entries:
            row=dict(video_id=str(item['video_id']),label=label,label_id=label_id,
                     split='validation' if item['split']=='val' else item['split'],
                     signer_id=item['signer_id'])
            output=args.output/(row['video_id']+'.npz')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            try:
                if output.exists():
                    with np.load(output,allow_pickle=False) as old:
                        cached=bool(str(old['feature_version'])==WORD_FEATURE_VERSION
                                and str(old['video_sha256'])==digest
                                and old['features'].shape==(SEQUENCE_FRAMES,WORD_FEATURE_DIM)
                                and np.isfinite(old['features']).all())
                        if cached:
                            features=old['features'].copy()
                            info=json.loads(str(old['decode_info']))
                    if not cached:features,timestamps,info=extract_clip(path,extractor)
                else:
                    cached=False;features,timestamps,info=extract_clip(path,extractor)
                if not cached:
                    np.savez_compressed(output,features=features,timestamps=timestamps,
                        feature_version=WORD_FEATURE_VERSION,video_sha256=digest,
                        decode_info=json.dumps(info))
                row.update(status='extracted',cached=cached,video_sha256=digest,
                           feature_path=str(output.resolve()),**info,**quality(features))
            except (ValueError,OSError,KeyError) as exc:
                row.update(status='failed',error=str(exc),quality_pass=False)
            rows.append(row)
            if len(rows)%10==0:print(f'Extracted {len(rows)}/{len(entries)}; quality passes '+str(sum(r['quality_pass'] for r in rows)),flush=True)
    finally:extractor.close()
    report=dict(feature_version=WORD_FEATURE_VERSION,feature_dim=WORD_FEATURE_DIM,
        sequence_frames=SEQUENCE_FRAMES,metadata_sha256=hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        processed=len(rows),status_counts=dict(Counter(r['status'] for r in rows)),
        quality_passes=sum(r['quality_pass'] for r in rows),
        quality_thresholds=dict(min_body_coverage=MIN_BODY_COVERAGE,min_any_hand_coverage=MIN_HAND_COVERAGE),
        rows=rows,note='Fixed-time samples, explicit masks, no interpolation of missing landmarks. '
        'Quality thresholds are provisional engineering gates, not signing correctness or accuracy. '
        'Pseudo-3D features are not calibrated metric coordinates. No training.')
    (args.output/'extraction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
