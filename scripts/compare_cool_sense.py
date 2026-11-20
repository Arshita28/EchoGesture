"""Evaluate two additional WLASL cool-neat templates against the active model."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from asl_lab.word_dense import extract_dense
from asl_lab.word_features import WordExtractor
from asl_lab.word_model import WordTemplateClassifier, classify_word
from scripts.train_word_background import load, metrics


def main():
    root = Path('datasets/raw/cool-sense')
    active = joblib.load('models/words-prototype.joblib')
    features = {}
    hashes = {}
    extractor = WordExtractor()
    try:
        for name in ['13205', '13206', 'ziggy-evaluation']:
            path = root / (name + '.mp4')
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            features[name] = extract_dense(path, extractor)[0]
    finally:
        extractor.close()

    manifests = ['word-dense', 'word-negative-dense']
    rows = []
    for folder in manifests:
        rows.extend(json.loads(Path('datasets/preprocessed', folder, 'extraction.json').read_text())['rows'])
    if len(set(hashes.values())) != 3 or set(hashes.values()) & {r.get('video_sha256') for r in rows}:
        raise ValueError('Duplicate clip')
    old = active['model']
    new = [features[k] for k in ['13205', '13206']]
    for x in new:
        if x[:, -1].mean() < .7 or (x[:, -3:-1].max(axis=1) > .5).mean() < .5:
            raise ValueError('Insufficient training clip visibility')
    model = WordTemplateClassifier().fit(np.concatenate([old.templates_, new]), np.r_[old.labels_, ['cool', 'cool']])
    candidate = dict(active, model=model, name='cool_neat_extension')
    joblib.dump(candidate, 'models/words-cool-neat-candidate.joblib')
    val = [r for r in rows if r['split'] == 'validation' and r['quality_pass']]
    truth = [r['label'] if r['label'] in active['labels'] else 'NONE' for r in val]
    report = {'sha256': hashes, 'training_ids': ['13205', '13206'], 'evaluation_source': 'https://www.ziggyasl.com/sign/cool'}
    for key, bundle in [('active', active), ('candidate', candidate)]:
        pred = bundle['model'].predict([load(r) for r in val])
        report[key] = {'validation': metrics(truth, pred),
                       'ziggy': classify_word(bundle, features['ziggy-evaluation'])}
    Path('reports/word-cool-neat.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
