"""Evaluate a frozen word prototype on newly acquired clips and fixed stress cases."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time
import cv2
import joblib
import numpy as np
from asl_lab.word_features import WordExtractor,WORD_FEATURE_VERSION
from asl_lab.word_model import classify_word
from scripts.extract_words import selected_indices


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_frames(path):
    cap=cv2.VideoCapture(str(path));frames=[]
    try:
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        if not cap.isOpened() or not 0<fps<=120:raise ValueError('Invalid FPS or cannot decode')
        while True:
            ok,frame=cap.read()
            if not ok:break
            h,w=frame.shape[:2]
            if max(h,w)>640:frame=cv2.resize(frame,(round(w*640/max(h,w)),round(h*640/max(h,w))))
            frames.append(frame)
            if len(frames)/fps>10:raise ValueError('Clip exceeds ten-second scope; requires explicit reviewed segment')
        if len(frames)<8:raise ValueError('Too few source frames')
        return frames,fps
    finally:cap.release()


def stress_frames(frames,fps,condition):
    if condition=='leading_hold_1s':return [frames[0]]*round(fps)+frames
    if condition=='trailing_hold_1s':return frames+[frames[-1]]*round(fps)
    if condition=='both_holds_1s':return [frames[0]]*round(fps)+frames+[frames[-1]]*round(fps)
    if condition=='original':return frames
    result=[]
    for frame in frames:
        h,w=frame.shape[:2]
        if condition=='dim_55_percent':f=np.clip(frame.astype(float)*.55,0,255).astype(np.uint8)
        elif condition=='low_resolution_160':
            f=cv2.resize(cv2.resize(frame,(160,max(1,round(h*160/w)))),(w,h))
        elif condition=='zoom_out_70_percent':
            f=np.zeros_like(frame);small=cv2.resize(frame,(round(w*.7),round(h*.7)))
            y=(h-small.shape[0])//2;x=(w-small.shape[1])//2;f[y:y+small.shape[0],x:x+small.shape[1]]=small
        elif condition=='shift_right_12_percent':
            f=cv2.warpAffine(frame,np.float32([[1,0,w*.12],[0,1,0]]),(w,h))
        elif condition=='roll_12_degrees':
            f=cv2.warpAffine(frame,cv2.getRotationMatrix2D((w/2,h/2),12,1),(w,h))
        else:raise ValueError('Unknown evaluation condition')
        result.append(f)
    return result


def summary(rows):
    complete=[r for r in rows if r['status']!='failed']
    matched=[r for r in complete if r['status']=='prototype_match']
    return dict(attempted=len(rows),processed=len(complete),failures=len(rows)-len(complete),
        correct=sum(r.get('correct',False) for r in complete),
        all_attempt_accuracy=sum(r.get('correct',False) for r in complete)/len(rows) if rows else None,
        detection_abstentions=sum(r['status']=='insufficient_detection' for r in complete),
        match_coverage=len(matched)/len(rows) if rows else None,
        accuracy_among_matches=sum(r['correct'] for r in matched)/len(matched) if matched else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config/word_external_evaluation.json')
    p.add_argument('--output',type=Path,default=Path('datasets/preprocessed/word-external-evaluation'))
    a=p.parse_args();config=json.loads(Path(a.config).read_text());a.output.mkdir(parents=True,exist_ok=True)
    model_hash=digest(config['model']);bundle=joblib.load(config['model'])
    if bundle['feature_version']!=WORD_FEATURE_VERSION:raise ValueError('Feature version mismatch')
    sources=json.loads(Path(config['sources']).read_text())
    used=json.loads(Path('datasets/preprocessed/words/quality-manifest.json').read_text())
    used=[r for r in used if r['label'] in bundle['labels']]
    used_hashes={r['video_sha256'] for r in used};used_ids={r['video_id'] for r in used}
    metadata=json.loads(Path('datasets/raw/wlasl/WLASL_v0.3.json').read_text())
    used_urls={i['url'] for e in metadata for i in e['instances'] if str(i['video_id']) in used_ids}
    seen=set();rows=[];source_audit=[];extractor=WordExtractor()
    try:
        for source in sources:
            record={k:source[k] for k in ('label','video_id','url','source_page','source')}
            if 'segment' in source:record['segment']=source['segment']
            if source['status']!='downloaded':
                source_audit.append({**record,'status':'download_failed'});continue
            sha=digest(source['path'])
            if sha in used_hashes or sha in seen or source['url'] in used_urls:
                source_audit.append({**record,'status':'overlap_excluded'});continue
            seen.add(sha)
            try:frames,fps=read_frames(source['path'])
            except (ValueError,OSError) as exc:
                source_audit.append({**record,'status':'decode_or_scope_failed','error':str(exc)});continue
            source_audit.append({**record,'status':'included','sha256':sha,'native_frames':len(frames),'fps':fps})
            for condition in config['conditions']:
                row=dict(video_id=source['video_id'],expected=source['label'],condition=condition)
                started=time.perf_counter()
                try:
                    changed=stress_frames(frames,fps,condition);indices=selected_indices(len(changed))
                    extractor.model.reset()
                    features=np.stack([extractor.extract(cv2.cvtColor(changed[i],cv2.COLOR_BGR2RGB)) for i in indices])
                    prediction=classify_word(bundle,features)
                    row.update(prediction);row.update(correct=prediction['label']==source['label'],
                        processing_ms=round((time.perf_counter()-started)*1000,2))
                    np.savez_compressed(a.output/(source['video_id']+'-'+condition+'.npz'),
                        features=features,source_sha256=sha,feature_version=WORD_FEATURE_VERSION)
                except (ValueError,OSError) as exc:row.update(status='failed',error=str(exc),correct=False)
                rows.append(row)
            (a.output/'trials.partial.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(source['label'],source['video_id'],'completed',len(config['conditions']),'conditions',flush=True)
    finally:extractor.close()
    if digest(config['model'])!=model_hash:raise ValueError('Model changed during evaluation')
    result=dict(model_sha256=model_hash,config_sha256=digest(a.config),feature_version=WORD_FEATURE_VERSION,
        source_audit=source_audit,by_condition={c:summary([r for r in rows if r['condition']==c]) for c in config['conditions']},
        by_word_original={label:summary([r for r in rows if r['expected']==label and r['condition']=='original']) for label in bundle['labels']},
        original_confusions=dict(Counter(r['expected']+' -> '+str(r.get('label')) for r in rows if r['condition']=='original')),
        rows=rows,policy=config['policy'],
        limitations='Convenience web sample with dictionary-provided labels. Sign variants and senses can differ. '
        'Exact hashes and known source URLs were checked against all fitting/selection/evaluation clips. '
        'Distinct video does not prove distinct signer; no claim of signer-independent sampling. '
        'Synthetic conditions share source clips, so do not treat their total as independent sample size. '
        'No model fitting or threshold changes. All newly evaluated clips must remain outside future fit data unless explicitly reclassified as development data.')
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    report={k:v for k,v in result.items() if k!='rows'}
    Path('reports/word-external-evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(result['by_condition'],indent=2))


if __name__=='__main__':main()
