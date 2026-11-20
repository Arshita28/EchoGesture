"""Calibrate a conservative rejection candidate on known/other-word validation only."""
import hashlib,json
from pathlib import Path
import joblib,numpy as np
from asl_lab.word_features import WordExtractor
from asl_lab.word_dense import extract_dense,DENSE_VERSION
from scripts.extract_words import quality


def main():
    modelpath=Path('models/words-prototype.joblib');bundle=joblib.load(modelpath);model=bundle['model']
    known=json.loads(Path('datasets/preprocessed/word-dense/extraction.json').read_text())['rows']
    known=[r for r in known if r['split']=='validation' and r['status']=='extracted' and r['quality_pass']]
    old=json.loads(Path('datasets/preprocessed/words/quality-manifest.json').read_text())
    negatives=[r for r in old if r['split']=='validation' and r['label'] not in bundle['labels']]
    out=Path('datasets/preprocessed/word-rejection');out.mkdir(parents=True,exist_ok=True)
    rows=[];extractor=WordExtractor()
    try:
        for r in known+negatives:
            if r in known:
                with np.load(r['feature_path'],allow_pickle=False) as f:x=f['features']
            else:
                cache=out/(r['video_id']+'.npz')
                if cache.exists():
                    with np.load(cache,allow_pickle=False) as f:x=f['features']
                else:
                    try:x,t,info=extract_dense(Path('datasets/raw/wlasl/videos')/(r['video_id']+'.mp4'),extractor)
                    except ValueError:x=np.zeros((32,154),np.float32)
                    np.savez_compressed(cache,features=x,preprocessing=DENSE_VERSION)
            q=quality(x);scores=model.decision_function(x.reshape(1,-1))[0];order=np.argsort(scores)[::-1]
            rows.append(dict(video_id=r['video_id'],expected=r['label'],known=r['label'] in bundle['labels'],
                predicted=str(model.classes_[order[0]]),distance=float(-scores[order[0]]),
                gap=float(scores[order[0]]-scores[order[1]]),quality_pass=q['quality_pass']))
    finally:extractor.close()
    candidates=[]
    # Predeclared small grid; no inspected/fresh web scores participate.
    for distance in [.12,.18,.25,.35,.5,1e6]:
        for gap in [0,.005,.015,.03,.06]:
            accept=lambda r:r['quality_pass'] and r['distance']<=distance and r['gap']>=gap
            kp=[r for r in rows if r['known']];np_=[r for r in rows if not r['known']]
            correct=sum(accept(r) and r['predicted']==r['expected'] for r in kp)
            rejected=sum(not accept(r) for r in np_)
            candidates.append(dict(max_distance=distance,min_gap=gap,known_correct_accepted=correct,
                known_total=len(kp),known_abstained=sum(not accept(r) for r in kp),
                unsupported_rejected=rejected,unsupported_total=len(np_),
                eligible=correct>=6 and rejected>=4))
    eligible=[r for r in candidates if r['eligible']]
    selected=max(eligible,key=lambda r:(r['known_correct_accepted']/r['known_total']+r['unsupported_rejected']/r['unsupported_total'],-r['known_abstained'])) if eligible else None
    if selected:
        candidate={**bundle,'name':bundle['name']+'-rejection-candidate','rejection':dict(max_distance=selected['max_distance'],min_gap=selected['min_gap'])}
        joblib.dump(candidate,'models/words-rejection-candidate.joblib')
    report=dict(base_model_sha256=hashlib.sha256(modelpath.read_bytes()).hexdigest(),selected=selected,
        candidates=candidates,validation_rows=rows,production_replaced=False,
        note='Calibration uses eight known and eight other-word validation clips only. Unknown words are a convenience sample; '
        'thresholds are heuristic distances, not probabilities. No idle rejection guarantee and no fresh-web tuning.')
    Path('reports/word-rejection-calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Selected:',selected,flush=True)

if __name__=='__main__':main()
