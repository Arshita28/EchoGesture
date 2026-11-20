"""Compare regularized ensembles on training groups and original validation.

No inspected web clips participate in fitting or candidate selection.
"""
import hashlib,json
from collections import Counter
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from asl_lab.word_summary import WordSummary
from asl_lab.word_ensemble import RankedClassifier
from asl_lab.word_features import WORD_FEATURE_VERSION
from asl_lab.word_dense import DENSE_VERSION
from scripts.train_word_background import load,metrics


def main():
    labels=json.loads(Path('config/word_prototype.json').read_text())['labels']
    known=json.loads(Path('datasets/preprocessed/word-dense/extraction.json').read_text())['rows']
    negatives=json.loads(Path('datasets/preprocessed/word-negative-dense/extraction.json').read_text())['rows']
    rows=[{**r,'target':r['label'] if r['label'] in labels else 'NONE'} for r in known+negatives if r['status']=='extracted' and r['quality_pass'] and r['split'] in ['train','validation']]
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
    x=np.array([load(r) for r in train]);y=np.array([r['target'] for r in train]);groups=np.array([str(r['signer_id']) for r in train])
    vx=np.array([load(r) for r in val]);vy=np.array([r['target'] for r in val])
    candidates={
        'summary_extra_trees':RankedClassifier(make_pipeline(WordSummary(),ExtraTreesClassifier(n_estimators=400,class_weight='balanced',min_samples_leaf=2,max_features=.3,random_state=42,n_jobs=-1))),
        'summary_random_forest':RankedClassifier(make_pipeline(WordSummary(),RandomForestClassifier(n_estimators=300,class_weight='balanced',max_depth=6,min_samples_leaf=2,random_state=42,n_jobs=-1))),
        'pca_rbf':make_pipeline(StandardScaler(),PCA(n_components=.95,svd_solver='full'),SVC(C=1,class_weight='balanced',decision_function_shape='ovr')),
        'summary_lda':RankedClassifier(make_pipeline(WordSummary(),StandardScaler(),PCA(n_components=.95,svd_solver='full'),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'))),
    }
    results={};fitted={}
    folds=list(GroupKFold(n_splits=3).split(x,y,groups))
    for name,model in candidates.items():
        scores=[]
        for fit_index,eval_index in folds:
            if len(set(y[fit_index]))<3:continue
            estimator=clone(model).fit(x[fit_index],y[fit_index]);truth=y[eval_index];pred=estimator.predict(x[eval_index])
            scores.append(dict(accuracy=float((pred==truth).mean()),classes_missing_from_fold_training=sorted(set(y)-set(y[fit_index])),samples=len(truth)))
        model.fit(x,y);fitted[name]=model;result=metrics(vy,model.predict(vx));result['grouped_training_cv']=scores
        result['grouped_cv_accuracy']=sum(s['accuracy']*s['samples'] for s in scores)/sum(s['samples'] for s in scores)
        results[name]=result;print(name,result,flush=True)
    active=joblib.load('models/words-prototype.joblib');baseline=metrics(vy,active['model'].predict(vx))
    eligible=[name for name,r in results.items() if r['known_correct']>=baseline['known_correct'] and r['other_rejected']>=baseline['other_rejected']]
    winner=max(eligible or list(results),key=lambda n:(results[n]['balanced_utility'],results[n]['grouped_cv_accuracy']))
    bundle=dict(name='five_word_ensemble_candidate_v5',labels=labels,model=fitted[winner],feature_version=WORD_FEATURE_VERSION,preprocessing=DENSE_VERSION,
        scope='Experimental ensemble with other-word class',unknown_handling='Limited pooled other-word class; not universal open-set rejection')
    path=Path('models/words-ensemble-candidate.joblib');joblib.dump(bundle,path)
    report=dict(selected=winner,eligible=bool(eligible),selection='Known and other-word validation non-regression required, then balanced utility and grouped training CV tie break. No web selection.',
        baseline_validation=baseline,candidates=results,training_count=len(train),validation_count=len(val),
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_bytes=path.stat().st_size,
        production_replaced=False,limitations='Tiny original validation repeatedly inspected; CV training folds can miss classes and those are reported. This is model development, not independent evidence of high accuracy.')
    Path('reports/word-ensemble-comparison.json').write_text(json.dumps(report,indent=2)+'\n');print('Selected',winner,'eligible',bool(eligible))

if __name__=='__main__':main()
