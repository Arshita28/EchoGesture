"""Compare dense six-class candidates with real other-word training negatives.

Selection uses known/other-word validation only. Inspected web sets are kept
as diagnostic regressions and never used to fit or select a candidate.
"""
import hashlib,json
from collections import Counter
from pathlib import Path
import joblib,numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,LinearSVC
from asl_lab.word_model import WordTemplateClassifier
from asl_lab.word_summary import WordSummary
from asl_lab.word_features import WORD_FEATURE_VERSION
from asl_lab.word_dense import DENSE_VERSION


def load(row):
    with np.load(row['feature_path'],allow_pickle=False) as f:
        if str(f['video_sha256'])!=row['video_sha256']:raise ValueError('Feature/source hash mismatch')
        if str(f['preprocessing'])!=DENSE_VERSION:raise ValueError('Inconsistent dense preprocessing')
        x=f['features']
        if x.shape!=(32,154) or not np.isfinite(x).all():raise ValueError('Invalid dense features')
        return x.reshape(-1)


def metrics(truth,prediction):
    truth=np.asarray(truth);prediction=np.asarray(prediction);known=truth!='NONE';other=~known
    if not known.any() or not other.any():raise ValueError('Both known and negative examples required')
    correct=int((prediction[known]==truth[known]).sum());rejected=int((prediction[other]=='NONE').sum())
    return dict(known_correct=correct,known_total=int(known.sum()),
        known_rejected=int((prediction[known]=='NONE').sum()),
        other_rejected=rejected,other_total=int(other.sum()),
        balanced_utility=(correct/known.sum()+rejected/other.sum())/2,
        macro_f1=float(f1_score(truth,prediction,average='macro',zero_division=0)))


def main():
    labels=json.loads(Path('config/word_prototype.json').read_text())['labels']
    known=json.loads(Path('datasets/preprocessed/word-dense/extraction.json').read_text())['rows']
    negatives=json.loads(Path('datasets/preprocessed/word-negative-dense/extraction.json').read_text())['rows']
    rows=[];seen={}
    for row in known+negatives:
        if row['status']!='extracted' or not row['quality_pass'] or row['split'] not in ['train','validation']:continue
        digest=row['video_sha256']
        if digest in seen:raise ValueError('Duplicate source across dataset: '+row['video_id'])
        seen[digest]=row['split'];rows.append({**row,'target':row['label'] if row['label'] in labels else 'NONE'})
    subsets={}
    for split in ['train','validation']:
        group=[r for r in rows if r['split']==split]
        if set(r['target'] for r in group)!=set(labels+['NONE']):raise ValueError('All six classes must be represented')
        subsets[split]=(np.array([load(r) for r in group]),np.array([r['target'] for r in group]))
    candidates={
        'dense_templates':WordTemplateClassifier(),
        'summary_linear_c01':make_pipeline(WordSummary(),StandardScaler(),LinearSVC(C=.1,class_weight='balanced',random_state=42)),
        'summary_linear_c1':make_pipeline(WordSummary(),StandardScaler(),LinearSVC(C=1,class_weight='balanced',random_state=42)),
        'summary_rbf_c1':make_pipeline(WordSummary(),StandardScaler(),SVC(C=1,class_weight='balanced',decision_function_shape='ovr')),
        'summary_rbf_c10':make_pipeline(WordSummary(),StandardScaler(),SVC(C=10,class_weight='balanced',decision_function_shape='ovr')),
    }
    results={}
    for name,model in candidates.items():
        model.fit(*subsets['train']);prediction=model.predict(subsets['validation'][0])
        results[name]=metrics(subsets['validation'][1],prediction)
        print(name,results[name],flush=True)
    baseline=joblib.load('models/words-prototype.joblib')
    old_prediction=baseline['model'].predict(subsets['validation'][0]);old=metrics(subsets['validation'][1],old_prediction)
    eligible=[name for name,r in results.items() if r['known_correct']>=6 and r['other_rejected']>=4]
    winner=max(eligible or list(results),key=lambda name:results[name]['balanced_utility'])
    bundle=dict(name='five_word_other_sign_candidate_v4',labels=labels,model=candidates[winner],
        feature_version=WORD_FEATURE_VERSION,preprocessing=DENSE_VERSION,
        scope='Five-word candidate with a pooled other-sign class; not a full sign-language translator.',
        unknown_handling='Other-word class learned from limited examples; no general open-set or idle guarantee.')
    path=Path('models/words-background-candidate.joblib');joblib.dump(bundle,path)
    report=dict(selected=winner,eligible_for_further_evaluation=bool(eligible),selection='Balanced known correctness and negative rejection on original validation only; known >=6/8, other rejected >=4/8 eligibility floor.',
        baseline_validation=old,candidates=results,
        training_counts=dict(Counter(subsets['train'][1])),validation_counts=dict(Counter(subsets['validation'][1])),
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_bytes=path.stat().st_size,
        production_replaced=False,
        limitations='Tiny validation repeatedly inspected, overlapping signers and pooled other-word class. No idle training clips. Fresh dictionary set not used in fit/selection. No live accuracy claim.')
    Path('reports/word-background-training.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
