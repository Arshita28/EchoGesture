"""Rank active vocabulary on cached ASL Citizen validation/test regression clips.

Selection reuses inspected data: scores are development evidence, not an
independent estimate of a selected/retrained 50-word model's accuracy.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from asl_lab.word_svm_features import summarize_batch
from scripts.extract_words import quality


def main():
    path=Path('models/word81-grammar-candidate.joblib')
    model=joblib.load(path)
    labels=list(map(str,model.classes_))
    root=Path('datasets/raw/asl-citizen-selected')
    results=[]
    for split in ('val','test'):
        records=[];features=[];indices=[]
        for row in csv.DictReader((root/(split+'.csv')).open()):
            label=row['Gloss'].lower()
            if label not in labels:continue
            record=dict(split=split,label=label,video=row['Video file'],signer=row['Participant ID'],prediction=None,correct=False,status='missing_features')
            cache=(root/row['Video file']).with_suffix('.npy')
            if cache.exists():
                x=np.load(cache,allow_pickle=False)
                if x.shape==(32,154) and np.isfinite(x).all() and quality(x)['quality_pass']:
                    record['status']='processed';indices.append(len(records));features.append(x)
                else:record['status']='insufficient_detection'
            records.append(record)
        predictions=model.predict(summarize_batch(features))
        for i,prediction in zip(indices,predictions):
            records[i].update(prediction=str(prediction),correct=str(prediction)==records[i]['label'])
        results.extend(records)
        print(split,sum(r['correct'] for r in records),'/',len(records),flush=True)
    def metric(rows):
        n=len(rows);k=sum(r['correct'] for r in rows)
        return dict(correct=k,total=n,accuracy=k/n if n else None,failures=sum(r['status']!='processed' for r in rows))
    ranking=[]
    for label in labels:
        groups={s:metric([r for r in results if r['label']==label and r['split']==s]) for s in ('val','test')}
        # Require both splits and meaningful test support; compare equal-weight
        # per-split accuracy so the larger test partition does not hide weak val.
        score=(groups['val']['accuracy']+groups['test']['accuracy'])/2 if groups['val']['total'] and groups['test']['total']>=8 else None
        confusions=Counter(r['prediction'] or r['status'] for r in results if r['label']==label and not r['correct'])
        ranking.append(dict(word=label,validation=groups['val'],test=groups['test'],selection_score=score,confusions=dict(confusions)))
    ranking.sort(key=lambda r:(r['selection_score'] is not None,r['selection_score'] or 0,r['test']['accuracy'] or 0,r['test']['total'],r['word']),reverse=True)
    selected=[r['word'] for r in ranking[:50]]
    assert len(selected)==50 and all(r['selection_score'] is not None for r in ranking[:50])
    report=dict(model=str(path),model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),method='Rank mean of validation and test all-attempt accuracy; require validation coverage and >=8 test attempts; ties: test accuracy, test support, word descending.',limitations=['Previously inspected cached ASL Citizen data, not fresh or live testing.','Selection uses these scores; selected-subset scores are selection-biased.','81-class model remains unchanged; no 50-class model was fitted.','Cached landmark replay includes quality rejection but does not rerun video decoding or live tracking.','No independent cross-source evidence for every word.'],selected=sorted(selected),all_words=ranking,overall={s:metric([r for r in results if r['split']==s]) for s in ('val','test')},selected_subset={s:metric([r for r in results if r['split']==s and r['label'] in selected]) for s in ('val','test')},attempts=results)
    Path('reports/showcase-word-ranking.json').write_text(json.dumps(report,indent=2)+'\n')
    with Path('reports/showcase-word-ranking.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['rank','word','recommended','validation_correct','validation_total','test_correct','test_total','selection_score','failures'])
        for i,r in enumerate(ranking,1):writer.writerow([i,r['word'],i<=50,r['validation']['correct'],r['validation']['total'],r['test']['correct'],r['test']['total'],r['selection_score'],r['validation']['failures']+r['test']['failures']])
    print(json.dumps({k:report[k] for k in ('selected','overall','selected_subset')},indent=2))
    print('UNCOVERED', [r['word'] for r in ranking if r['selection_score'] is None])
    print('SELECTED RESULTS',[(r['word'],str(r['test']['correct'])+'/'+str(r['test']['total'])) for r in ranking[:50]])

if __name__=='__main__':main()
