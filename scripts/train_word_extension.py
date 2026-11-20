"""Compare expanded training-only candidates without replacing the active model."""
import hashlib,json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from asl_lab.word_candidate import MaskedWordClassifier
from asl_lab.word_model import WordTemplateClassifier,classify_word
from asl_lab.word_features import WORD_FEATURE_VERSION
from scripts.extract_words import quality


def read(row):
    with np.load(row['feature_path'],allow_pickle=False) as f:return f['features'].reshape(-1)


def main():
    labels=json.loads(Path('config/word_prototype.json').read_text())['labels']
    original=json.loads(Path('datasets/preprocessed/words/quality-manifest.json').read_text())
    extension=json.loads(Path('datasets/preprocessed/word-training-extension/extraction.json').read_text())['rows']
    original_hashes={r['video_sha256'] for r in original}
    rows=[];rejected=[]
    for row in extension:
        if row['split']!='train' or row['label'] not in labels:raise ValueError('Extension is training-only declared vocabulary')
        if row['status']!='extracted' or not row['quality_pass']:rejected.append(row['video_id']);continue
        if row['video_sha256'] in original_hashes:raise ValueError('Duplicate source content')
        original_hashes.add(row['video_sha256']);rows.append(row)
    train=[r for r in original if r['label'] in labels and r['split']=='train']+rows
    val=[r for r in original if r['label'] in labels and r['split']=='validation']
    test=[r for r in original if r['label'] in labels and r['split']=='test']
    negative=[r for r in original if r['label'] not in labels and r['split']=='train']
    models={};results={}
    for name,model,fit in [('expanded_templates',WordTemplateClassifier(),train),
        ('expanded_masked',MaskedWordClassifier(1.),train),
        ('expanded_masked_other_words',MaskedWordClassifier(1.),train+negative)]:
        model.fit(np.array([read(r) for r in fit]),[r['label'] if r['label'] in labels else 'NONE' for r in fit])
        pred=model.predict(np.array([read(r) for r in val]));expected=[r['label'] for r in val]
        results[name]=dict(validation_correct=int((pred==expected).sum()),validation_total=len(val),
            validation_macro_f1=float(f1_score(expected,pred,average='macro',zero_division=0)))
        models[name]=model
    winner=max(results,key=lambda k:results[k]['validation_macro_f1']);model=models[winner]
    path=Path('models/words-expanded-candidate.joblib')
    joblib.dump(dict(name='five_word_expanded_candidate',model=model,labels=labels,feature_version=WORD_FEATURE_VERSION,
        scope='Expanded training development candidate, not promoted',unknown_handling='Experimental, not validated open-set rejection'),path)
    regression={}
    for title,group in [('historical_test',test)]:
        pred=model.predict(np.array([read(r) for r in group]));regression[title]=dict(correct=int((pred==[r['label'] for r in group]).sum()),total=len(group))
    trials=json.loads(Path('datasets/preprocessed/word-external-evaluation/results.json').read_text())['rows']
    regressions=[]
    for row in trials:
        with np.load(Path('datasets/preprocessed/word-external-evaluation')/(row['video_id']+'-'+row['condition']+'.npz'),allow_pickle=False) as f:features=f['features']
        inference=classify_word(joblib.load(path),features)
        prediction=inference['label'] if inference['status']=='prototype_match' else None
        regressions.append(dict(expected=row['expected'],condition=row['condition'],predicted=prediction,correct=prediction==row['expected']))
    for condition in sorted({r['condition'] for r in regressions}):
        group=[r for r in regressions if r['condition']==condition]
        regression[condition]=dict(correct=sum(r['correct'] for r in group),total=len(group),rejected=sum(r['predicted'] is None or r['predicted']=='NONE' for r in group))
    report=dict(selected=winner,selection='Original known-word validation macro F1 only; web data never fitted or used for selection.',
        candidates=results,added_quality_passing_training=len(rows),added_quality_rejected_ids=rejected,
        known_training_count=len(train),regressions=regression,model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        production_replaced=False,note='Historical test and inspected web clips are development regressions, not fresh accuracy evidence. '
        'No automatic promotion. Sparse original validation set and source/signer overlap remain limitations.')
    Path('reports/word-expanded-candidate.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
