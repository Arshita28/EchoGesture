"""Measure the broad pose descriptor using all locally cached exact ASL Citizen matches."""
import csv
import json
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

from asl_lab.broad_pose import from_cached
from scripts.extract_words import quality


root = Path('datasets/raw/asl-citizen-selected')
bank = {r['gloss'] for r in json.loads(Path('datasets/raw/wlasl/WLASL_v0.3.json').read_text())}
sets = {}
official_counts = {}
for role in ('train', 'val', 'test'):
    x, y, missed = [], [], []
    official_counts[role] = Counter()
    for row in csv.DictReader((root / f'{role}.csv').open()):
        path = (root / row['Video file']).with_suffix('.npy')
        word = row['Gloss'].lower()
        official_counts[role][word] += 1
        if word not in bank or not path.exists():
            continue
        seq = np.load(path)
        try:
            if not quality(seq)['quality_pass']:
                raise ValueError('Tracking quality')
            x.append(from_cached(seq))
            y.append(word)
        except ValueError:
            missed.append(word)
    sets[role] = (np.stack(x), np.asarray(y), missed)
x, y, _ = sets['train']
val_x, val_y, _ = sets['val']
validation = {}
best = -1
for kind, alpha in [('linear', 1.), ('rbf', .1), ('rbf', 1.)]:
    stages = [StandardScaler(), Normalizer()]
    if kind == 'rbf':
        stages.append(Nystroem(gamma=2., n_components=512, random_state=42))
    stages.append(RidgeClassifier(alpha=alpha, solver='cholesky'))
    candidate = make_pipeline(*stages).fit(x, y)
    correct = int(sum(candidate.predict(val_x) == val_y))
    name = f'{kind}/{alpha}'
    validation[name] = correct
    if correct > best:
        best, model, selected = correct, candidate, name
report = dict(classes=len(model.classes_), train=len(y), descriptor='cached MediaPipe baseline',
              validation_selection=validation, selected=selected)
for role in ('val', 'test'):
    x, y, missed = sets[role]
    report[role] = dict(correct=int(sum(model.predict(x) == y)), cached_attempts=len(y) + len(missed),
                        official_attempts=sum(official_counts[role][w] for w in model.classes_))
joblib.dump(model, 'models/broad-cache-baseline.joblib', compress=3)
Path('reports/broad-cache-baseline.json').write_text(json.dumps(report, indent=2))
print(report)
