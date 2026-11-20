"""Verify the preserved Phase 1 baseline on saved public features without fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from asl_lab.train import load_records
from asl_lab.features import FEATURE_VERSION


def verify(samples, model, reference):
    rows = load_records(samples)
    bundle = joblib.load(model)  # Only the trusted local baseline checkpoint.
    if bundle['feature_version'] != FEATURE_VERSION:
        raise ValueError('Baseline model feature version mismatch')
    expected = json.loads(Path(reference).read_text())
    if hashlib.sha256(Path(samples).read_bytes()).hexdigest() != expected['data_sha256']:
        raise ValueError('Features differ from the recorded public baseline')
    results = {}
    for split in ('validation', 'test'):
        subset = [r for r in rows if r['split'] == split]
        x = np.asarray([r['features'] for r in subset]); y = [r['label'] for r in subset]
        prediction = bundle['model'].predict(x)
        results[split] = dict(samples=len(y), accuracy=float(accuracy_score(y, prediction)),
                             macro_f1=float(f1_score(y, prediction, average='macro')))
    selected = expected['models'][expected['selected']]
    checks = {
        'validation_macro_f1_matches': abs(results['validation']['macro_f1'] - selected['validation_macro_f1']) < 1e-10,
        'test_accuracy_matches': abs(results['test']['accuracy'] - selected['test_accuracy']) < 1e-10,
        'test_macro_f1_matches': abs(results['test']['macro_f1'] - selected['test_macro_f1']) < 1e-10,
    }
    return dict(verified=all(checks.values()), checks=checks, results=results,
        model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),
        feature_sha256=expected['data_sha256'], labels=bundle['labels'],
        model_name=bundle['name'], threshold=bundle['threshold'],
        note='No fitting, threshold changes, or artifact replacement. Reproduction of the historical '
             'static public pilot only; not an independent evaluation of the active personalized model '
             'or a measurement of live camera performance.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=Path('datasets/preprocessed/landmarks.jsonl'))
    parser.add_argument('--model', type=Path, default=Path('models/baseline-aspect-v2.joblib'))
    parser.add_argument('--reference', type=Path, default=Path('reports/evaluation.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/phase1-reproduction.json'))
    args = parser.parse_args()
    report = verify(args.samples, args.model, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['verified'] else 1)
