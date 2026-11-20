"""Train an explicitly declared word prototype; select on validation before test."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,LinearSVC
from asl_lab.word_model import WordTemplateClassifier
from asl_lab.word_features import WORD_FEATURE_VERSION
from scripts.audit_word_sequences import audit


def train(extraction,labels_file,config,model_path,report_path):
    specification=json.loads(Path(config).read_text());labels=specification['labels']
    if len(labels)<3 or len(set(labels))!=len(labels):raise ValueError('Declare at least three distinct prototype labels')
    full_labels=json.loads(Path(labels_file).read_text())
    integrity,rows=audit(extraction,full_labels)
    if integrity['errors']:raise ValueError('Sequence integrity audit failed')
    if any(label not in full_labels for label in labels):raise ValueError('Unknown prototype class')
    rows=[r for r in rows if r['label'] in labels]
    subsets={};counts={};groups=defaultdict(set)
    for split in ('train','validation','test'):
        selected=[r for r in rows if r['split']==split]
        counts[split]=dict(Counter(r['label'] for r in selected))
        if set(counts[split])!=set(labels):raise ValueError('Every declared word needs each official partition')
        if split=='train' and min(counts[split].values())<2:raise ValueError('Need at least two training clips per word')
        x=[]
        for row in selected:
            with np.load(row['feature_path'],allow_pickle=False) as sample:x.append(sample['features'].reshape(-1))
            groups[str(row['signer_id'])].add(split)
        subsets[split]=(np.asarray(x),np.asarray([r['label'] for r in selected]),selected)
    candidates={
        'linear_svm_c01':make_pipeline(StandardScaler(),LinearSVC(C=.1,class_weight='balanced',random_state=42)),
        'rbf_svm_c1':make_pipeline(StandardScaler(),SVC(C=1,class_weight='balanced',decision_function_shape='ovr')),
        'rbf_svm_c10':make_pipeline(StandardScaler(),SVC(C=10,class_weight='balanced',decision_function_shape='ovr')),
        'temporal_templates':WordTemplateClassifier(),
    }
    validation={}
    for name,model in candidates.items():
        model.fit(*subsets['train'][:2]);prediction=model.predict(subsets['validation'][0])
        validation[name]=dict(accuracy=float(accuracy_score(subsets['validation'][1],prediction)),
                             macro_f1=float(f1_score(subsets['validation'][1],prediction,average='macro',zero_division=0)))
    winner=max(candidates,key=lambda n:validation[n]['macro_f1'])
    model=candidates[winner]
    # Only the validation-selected winner sees test data. Do not tune after this.
    x,y,test_rows=subsets['test'];started=time.perf_counter();pred=model.predict(x)
    elapsed=(time.perf_counter()-started)*1000
    bundle=dict(name=specification['name'],model=model,labels=labels,feature_version=WORD_FEATURE_VERSION,
        scope=specification['scope'],unknown_handling=specification['unknown_handling'])
    model_path=Path(model_path);model_path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(bundle,model_path)
    report=dict(name=bundle['name'],scope=bundle['scope'],labels=labels,selected=winner,
        selection='Validation macro F1 only; deterministic candidate-order tie break; no refit on validation or test.',
        split_counts=counts,validation=validation,
        test=dict(clips=len(y),correct=int((pred==y).sum()),accuracy=float(accuracy_score(y,pred)),
            macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
            classification_report=classification_report(y,pred,labels=labels,output_dict=True,zero_division=0),
            confusion_matrix=confusion_matrix(y,pred,labels=labels).tolist(),
            predictions=[dict(video_id=r['video_id'],expected=str(a),predicted=str(b)) for r,a,b in zip(test_rows,y,pred)]),
        classifier_batch_ms=elapsed,model_bytes=model_path.stat().st_size,
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        extraction_sha256=hashlib.sha256(Path(extraction).read_bytes()).hexdigest(),
        signer_ids_crossing_partitions=sum(len(v)>1 for v in groups.values()),
        unknown_handling=bundle['unknown_handling'],
        limitations='Tiny available-data subset, selected for partition coverage before training. Official splits overlap signers. '
        'Metrics cover quality-passing clips only, not all attempted videos or live users. '
        'Unquantized desktop prototype, no trained background rejection. No edge latency or accuracy target compliance claim.')
    report_path=Path(report_path);report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extraction',default='datasets/preprocessed/words/extraction.json')
    p.add_argument('--labels',default='datasets/preprocessed/wlasl/labels.json')
    p.add_argument('--config',default='config/word_prototype.json')
    p.add_argument('--model',default='models/words-prototype.joblib')
    p.add_argument('--report',default='reports/word-prototype-training.json')
    a=p.parse_args();r=train(a.extraction,a.labels,a.config,a.model,a.report)
    print(json.dumps({k:r[k] for k in ['selected','split_counts','validation','test','model_bytes','limitations']},indent=2))
