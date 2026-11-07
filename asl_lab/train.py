"""Train using explicit signer/session-disjoint train/validation/test partitions."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .features import FEATURE_VERSION, LETTERS


def load_records(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError('No landmark samples found')
    for r in records:
        x = np.asarray(r['features'])
        if x.shape != (63,) or not np.isfinite(x).all() or r['label'] not in LETTERS:
            raise ValueError('Invalid features or unsupported label')
        if r.get('feature_version') != FEATURE_VERSION:
            raise ValueError('Feature version mismatch; re-extract data')
        if not r.get('group') or r.get('split') not in ('train', 'validation', 'test'):
            raise ValueError('Every sample needs a verified group and split')
    groups, hashes = {}, {}
    for r in records:
        for value, seen in ((r['group'], groups), (r['sha256'], hashes)):
            if value in seen and seen[value] != r['split']:
                raise ValueError('Group or duplicate image crosses partitions')
            seen[value] = r['split']
    labels = set(r['label'] for r in records)
    for split in ('train', 'validation', 'test'):
        if set(r['label'] for r in records if r['split'] == split) != labels:
            raise ValueError(f'{split} must contain every class')
    if len(labels) < 2:
        raise ValueError('Need at least two classes')
    return records


def make_candidates(sample_count):
    return {
        'svm': make_pipeline(StandardScaler(), SVC(C=10, kernel='rbf', probability=True, random_state=42)),
        'random_forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        'knn': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=min(5, sample_count), weights='distance')),
    }


def train(path, output=Path('models'), report_dir=Path('reports')):
    records = load_records(path)
    subsets = {}
    for split in ('train', 'validation', 'test'):
        rows = [r for r in records if r['split'] == split]
        subsets[split] = (np.array([r['features'] for r in rows]), np.array([r['label'] for r in rows]))
    x, y = subsets['train']
    candidates = make_candidates(len(x))
    results = {}
    for name, model in candidates.items():
        model.fit(x, y)
        pred = model.predict(subsets['validation'][0])
        results[name] = {'validation_macro_f1': float(f1_score(subsets['validation'][1], pred, average='macro'))}
    winner = max(results, key=lambda name: results[name]['validation_macro_f1'])
    model = candidates[winner]
    # Fixed before test evaluation. Scores are heuristic, not calibrated OOD detection.
    threshold = 0.70
    labels = sorted(set(y))
    for name, candidate in candidates.items():
        xt, yt = subsets['test']
        pred = candidate.predict(xt)
        scores = candidate.predict_proba(xt).max(axis=1)
        accepted = scores >= threshold
        results[name].update(test_accuracy=float(accuracy_score(yt, pred)),
            test_macro_f1=float(f1_score(yt, pred, average='macro')),
            coverage=float(accepted.mean()),
            accepted_accuracy=float(accuracy_score(yt[accepted], pred[accepted])) if accepted.any() else None,
            classification_report=classification_report(yt, pred, output_dict=True, zero_division=0),
            confusion_matrix=confusion_matrix(yt, pred, labels=labels).tolist())
    report = {'selected': winner, 'selection': 'validation macro F1 only', 'threshold': threshold,
              'labels': labels, 'models': results, 'feature_version': FEATURE_VERSION,
              'data_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              'split_groups': {s: sorted(set(r['group'] for r in records if r['split'] == s)) for s in subsets},
              'split_counts': {s: len(v[1]) for s, v in subsets.items()},
              'note': 'Static isolated signs only. SVM internal probability fitting is not group-aware; scores are heuristic.'}
    output, report_dir = Path(output), Path(report_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'threshold': threshold, 'feature_version': FEATURE_VERSION,
                 'labels': labels, 'name': winner}, output / 'classifier.joblib')
    (report_dir / 'evaluation.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('samples', nargs='?', default='data/processed/landmarks.jsonl')
    args = parser.parse_args()
    print(json.dumps(train(args.samples), indent=2))
