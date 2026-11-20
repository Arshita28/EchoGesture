"""Train dense-preprocessing candidates; validate before historical regression."""
import hashlib,json
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,LinearSVC
from asl_lab.word_model import WordTemplateClassifier
from asl_lab.word_candidate import MaskedWordClassifier
from asl_lab.word_summary import WordSummary
from asl_lab.word_dense import DENSE_VERSION
from asl_lab.word_features import WORD_FEATURE_VERSION


def load(row):
    with np.load(row['feature_path'],allow_pickle=False) as f:
        if str(f['preprocessing'])!=DENSE_VERSION:raise ValueError('Dense feature version mismatch')
        if str(f['video_sha256'])!=row['video_sha256']:raise ValueError('Source hash mismatch')
        return f['features'].reshape(-1)


def train():
    labels=json.loads(Path('config/word_prototype.json').read_text())['labels']
    extraction=json.loads(Path('datasets/preprocessed/word-dense/extraction.json').read_text())
    rows=[r for r in extraction['rows'] if r['status']=='extracted' and r['quality_pass']]
    seen={}
    for row in rows:
        if row['video_sha256'] in seen:raise ValueError('Duplicate video content')
        seen[row['video_sha256']]=row['split']
    subsets={};counts={}
    for split in ['train','validation','test']:
        part=[r for r in rows if r['split']==split];counts[split]=dict(Counter(r['label'] for r in part))
        if set(counts[split])!=set(labels):raise ValueError('Dense dataset missing partition classes')
        subsets[split]=(np.array([load(r) for r in part]),np.array([r['label'] for r in part]),part)
    candidates={
        'templates':WordTemplateClassifier(),
        'masked':MaskedWordClassifier(1),
        'summary_linear_c01':make_pipeline(WordSummary(),StandardScaler(),LinearSVC(C=.1,class_weight='balanced',random_state=42)),
        'summary_rbf_c1':make_pipeline(WordSummary(),StandardScaler(),SVC(C=1,class_weight='balanced',decision_function_shape='ovr')),
        'summary_rbf_c10':make_pipeline(WordSummary(),StandardScaler(),SVC(C=10,class_weight='balanced',decision_function_shape='ovr')),
    }
    results={}
    for name,model in candidates.items():
        model.fit(*subsets['train'][:2]);pred=model.predict(subsets['validation'][0]);truth=subsets['validation'][1]
        results[name]=dict(validation_macro_f1=float(f1_score(truth,pred,average='macro',zero_division=0)),
            validation_correct=int((pred==truth).sum()),validation_total=len(truth))
        print(name,results[name],flush=True)
    sparse=json.loads(Path('datasets/preprocessed/words/quality-manifest.json').read_text())
    matched_ids={r['video_id'] for r in sparse if r['label'] in labels and r['split']=='validation'}
    shared=[r for r in rows if r['video_id'] in matched_ids]
    for name,model in candidates.items():
        pred=model.predict(np.array([load(r) for r in shared]));truth=np.array([r['label'] for r in shared])
        results[name]['shared_validation_correct']=int((pred==truth).sum())
        results[name]['shared_validation_total']=len(shared)
    winner=max(results,key=lambda k:results[k]['validation_macro_f1']);model=candidates[winner]
    bundle=dict(name='five_word_dense_candidate_v3',model=model,labels=labels,feature_version=WORD_FEATURE_VERSION,
        preprocessing=DENSE_VERSION,scope='Dense isolated-sign candidate, not automatically promoted',unknown_handling='No trained idle or unknown rejection')
    path=Path('models/words-dense-candidate.joblib');joblib.dump(bundle,path)
    x,y,part=subsets['test'];pred=model.predict(x)
    report=dict(selected=winner,selection='Dense validation macro F1 only; historical test inspected only after selection.',
        split_counts=counts,candidates=results,historical_test=dict(correct=int((pred==y).sum()),total=len(y),
        predictions=[dict(video_id=r['video_id'],expected=str(a),predicted=str(b)) for r,a,b in zip(part,y,pred)]),
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_bytes=path.stat().st_size,
        preprocessing=DENSE_VERSION,production_replaced=False,
        limitations='Very small validation, overlapping signers, no fresh live evaluation; historical test is not untouched. '
        'Extraction success differs from sparse pipeline, so test denominators may differ. No background rejection or edge compliance claim.')
    Path('reports/word-dense-training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':train()
