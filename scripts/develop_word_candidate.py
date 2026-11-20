"""Develop mask-aware models using original training/validation only.

Historical test and inspected external clips are diagnostic regressions, not
fresh evaluation or selection criteria. No production artifact is replaced.
"""
import json,hashlib
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from asl_lab.word_candidate import MaskedWordClassifier
from asl_lab.word_features import WORD_FEATURE_VERSION


def load(row):
    with np.load(row['feature_path'],allow_pickle=False) as f:return f['features'].reshape(-1)


def main():
    labels=json.loads(Path('config/word_prototype.json').read_text())['labels']
    rows=json.loads(Path('datasets/preprocessed/words/quality-manifest.json').read_text())
    targets=[r for r in rows if r['label'] in labels]
    train=[r for r in targets if r['split']=='train'];val=[r for r in targets if r['split']=='validation']
    # Other labeled words are real negative examples, not fabricated idle signs.
    negative_train=[r for r in rows if r['label'] not in labels and r['split']=='train']
    negative_val=[r for r in rows if r['label'] not in labels and r['split']=='validation']
    fit=train+negative_train
    x=np.array([load(r) for r in fit]);y=np.array([r['label'] if r['label'] in labels else 'NONE' for r in fit])
    validation=val+negative_val;vx=np.array([load(r) for r in validation]);vy=np.array([r['label'] if r['label'] in labels else 'NONE' for r in validation])
    results={};models={}
    for weight in [.25,1.,4.]:
        name='masked_location_'+str(weight);model=MaskedWordClassifier(weight).fit(x,y)
        prediction=model.predict(vx);models[name]=model
        results[name]=dict(validation_macro_f1=float(f1_score(vy,prediction,average='macro',zero_division=0)),
            validation_accuracy=float(accuracy_score(vy,prediction)),
            known_correct=int((prediction[:len(val)]==vy[:len(val)]).sum()),known_total=len(val),
            other_word_rejections=int((prediction[len(val):]=='NONE').sum()),other_word_total=len(negative_val))
        print(name,results[name],flush=True)
    winner=max(models,key=lambda n:results[n]['validation_macro_f1']);model=models[winner]
    bundle=dict(name='five_word_masked_candidate_v2',labels=labels,model=model,feature_version=WORD_FEATURE_VERSION,
        scope='Development candidate, not production. Five words plus other-word negatives.',unknown_handling='Nearest other-word templates; not reliable open-set or idle rejection.')
    path=Path('models/words-masked-candidate.joblib');joblib.dump(bundle,path)
    # Inspect original test only after selection; never tune based on these results.
    test=[r for r in targets if r['split']=='test'];tx=np.array([load(r) for r in test]);pred=model.predict(tx)
    test_result=dict(correct=int((pred==[r['label'] for r in test]).sum()),total=len(test),
        predictions=[dict(video_id=r['video_id'],expected=r['label'],predicted=str(p)) for r,p in zip(test,pred)])
    # Already inspected web clips now function solely as a development regression.
    external=json.loads(Path('datasets/preprocessed/word-external-evaluation/results.json').read_text())
    by_condition={};external_rows=[]
    for row in external['rows']:
        p=Path('datasets/preprocessed/word-external-evaluation')/(row['video_id']+'-'+row['condition']+'.npz')
        if not p.exists():continue
        with np.load(p,allow_pickle=False) as f:features=f['features']
        prediction=str(model.predict(features.reshape(1,-1))[0])
        external_rows.append(dict(video_id=row['video_id'],condition=row['condition'],expected=row['expected'],predicted=prediction,correct=prediction==row['expected']))
    for condition in sorted({r['condition'] for r in external_rows}):
        group=[r for r in external_rows if r['condition']==condition]
        by_condition[condition]=dict(correct=sum(r['correct'] for r in group),total=len(group),rejected=sum(r['predicted']=='NONE' for r in group))
    report=dict(selection='Validation macro F1 only; no production promotion',selected=winner,validation=results,
        train_known=len(train),train_other_words=len(negative_train),validation_known=len(val),validation_other_words=len(negative_val),
        historical_test_regression=test_result,inspected_web_development_regression=by_condition,
        inspected_web_predictions=external_rows,model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        production_replaced=False,note='Neither historical test nor inspected web data is fresh evaluation. '
        'No web clip was fitted. Other-word rejection is not validated idle/open-set rejection. '
        'Mask-aware trim removes only outer missing-detection frames; internal gaps retained. '
        'Candidate uses existing sparse samples; additional independent data still required.')
    Path('reports/word-masked-candidate.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['inspected_web_predictions']},indent=2))

if __name__=='__main__':main()
