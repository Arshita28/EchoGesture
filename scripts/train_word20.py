"""Train a separate 20-label template benchmark using official dataset roles."""
import json
from pathlib import Path
from collections import Counter
import joblib
import numpy as np
from asl_lab.word_model import WordTemplateClassifier
from asl_lab.word_dense import DENSE_VERSION
from asl_lab.word_features import WORD_FEATURE_VERSION
from scripts.train_word_background import load


def main():
    labels=json.loads(Path('config/word20.json').read_text())['labels']
    rows=json.loads(Path('datasets/preprocessed/word20-dense/extraction.json').read_text())['rows']
    rows=[r for r in rows if r.get('quality_pass')]
    hashes=[r['video_sha256'] for r in rows]
    if len(hashes)!=len(set(hashes)):
        raise ValueError('Duplicate source hashes need review')
    train=[r for r in rows if r['split']=='train']
    missing=set(labels)-{r['label'] for r in train}
    if missing:raise ValueError('Missing training labels: '+str(sorted(missing)))
    model=WordTemplateClassifier().fit([load(r) for r in train],[r['label'] for r in train])
    report={'train_counts':dict(Counter(r['label'] for r in train)), 'evaluation':{}}
    for split in ['validation','test']:
        part=[r for r in rows if r['split']==split]
        pred=model.predict([load(r) for r in part])
        report['evaluation'][split]={'correct':int(sum(p==r['label'] for p,r in zip(pred,part))), 'total':len(part),
            'rows':[dict(video_id=r['video_id'],truth=r['label'],prediction=str(p)) for p,r in zip(pred,part)]}
    joblib.dump(dict(name='word20_template_candidate',model=model,labels=labels,feature_version=WORD_FEATURE_VERSION,preprocessing=DENSE_VERSION),'models/words20-candidate.joblib')
    Path('reports/word20-training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='evaluation'}))
    print({k:(v['correct'],v['total']) for k,v in report['evaluation'].items()})

if __name__=='__main__':main()
