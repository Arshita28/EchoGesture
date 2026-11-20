"""Frozen baseline/rejection comparison on a newly reserved dictionary batch."""
import argparse,hashlib,json
from pathlib import Path
import cv2,joblib,numpy as np
from asl_lab.word_features import WordExtractor
from asl_lab.word_dense import crop_sequence,DENSE_VERSION
from asl_lab.word_model import classify_word
from scripts.evaluate_words_external import read_frames,stress_frames


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',default='models/words-rejection-candidate.joblib')
    parser.add_argument('--candidate-name',default='rejection_candidate')
    parser.add_argument('--report',default='reports/word-fresh-dictionary-evaluation.json')
    args=parser.parse_args()
    if args.candidate_name=='active':parser.error('Candidate name must differ from active')
    paths={'active':Path('models/words-prototype.joblib'),args.candidate_name:Path(args.candidate)}
    hashes={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in paths.items()}
    bundles={name:joblib.load(p) for name,p in paths.items()}
    sources=json.loads(Path('datasets/raw/fresh-dictionary/sources.json').read_text())['sources']
    output=Path('datasets/preprocessed/fresh-dictionary');output.mkdir(parents=True,exist_ok=True)
    rows=[];excluded=[];seen=set();extractor=WordExtractor()
    try:
        for source in sources:
            if source['status']!='downloaded':excluded.append(source);continue
            if source['sha256'] in seen:excluded.append({**source,'reason':'duplicate_within_batch'});continue
            seen.add(source['sha256'])
            frames,fps=read_frames(source['path'])
            for condition in ['original','dim_55_percent','both_holds_1s']:
                cache=output/(source['video_id']+'-'+condition+'.npz')
                if cache.exists():
                    with np.load(cache,allow_pickle=False) as f:
                        if str(f['sha256'])!=source['sha256']:raise ValueError('Cached source hash mismatch')
                        x=f['features']
                else:
                    changed=stress_frames(frames,fps,condition);all_x=[];timestamps=[];next_time=0;extractor.model.reset()
                    for index,frame in enumerate(changed):
                        if index/fps+1e-7<next_time:continue
                        all_x.append(extractor.extract(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)));timestamps.append(index/fps);next_time+=1/25
                    try:x,t,info=crop_sequence(all_x,timestamps)
                    except ValueError:x=np.zeros((32,154),np.float32)
                    np.savez_compressed(cache,features=x,sha256=source['sha256'],preprocessing=DENSE_VERSION)
                for name,bundle in bundles.items():
                    result=classify_word(bundle,x);known=source['label'] in bundle['labels']
                    rows.append(dict(video_id=source['video_id'],source_url=source['url'],expected=source['label'],known=known,
                        condition=condition,model=name,predicted=result['label'],status=result['status'],
                        correct=(result['label']==source['label']) if known else result['label'] is None))
            (output/'trials.partial.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(source['label'],source['video_id'],'completed',flush=True)
    finally:extractor.close()
    summary={}
    for name in bundles:
        summary[name]={}
        for condition in ['original','dim_55_percent','both_holds_1s']:
            subset=[r for r in rows if r['model']==name and r['condition']==condition];known=[r for r in subset if r['known']];unknown=[r for r in subset if not r['known']]
            summary[name][condition]=dict(known_correct=sum(r['correct'] for r in known),known_total=len(known),known_abstained=sum(r['predicted'] is None for r in known),
                unsupported_rejected=sum(r['predicted'] is None for r in unknown),unsupported_total=len(unknown))
    for name,p in paths.items():
        if hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[name]:raise ValueError('Model changed during evaluation')
    report=dict(frozen_model_hashes=hashes,summary=summary,source_count=len(seen),sources=sources,excluded=excluded,rows=rows,
        policy='No fitting or threshold changes on this batch. Candidate calibrated on existing known/other-word validation before these predictions. '
        'When re-used after inspection, these are diagnostic regressions, not a fresh test. Dictionary-provided labels and name/hash exclusion, not guaranteed distinct signers or absence of re-encoded duplicates. '
        'Synthetic stress trials correlated; do not pool as independent samples. Unsupported probes are signs, not idle video.')
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
