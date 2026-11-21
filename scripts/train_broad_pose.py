"""Train a research classifier on the WLASL/ASL Citizen text-label intersection.

Uses ASL Citizen (Desai et al., NeurIPS 2023) with WholeBodyPose landmark data
(Lazo-Quispe et al., DOI 10.1117/12.3096312; CC BY-NC-SA 4.0).
"""
import csv
import hashlib
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

from asl_lab.broad_pose import BODY, HAND, describe

root = Path('datasets/raw/broad-pose')
citizen = Path('datasets/raw/asl-citizen-selected')
bank = {r['gloss'].lower() for r in json.loads(Path('datasets/raw/wlasl/WLASL_v0.3.json').read_text())}
# Verify the release before opening HDF5 files (including interrupted downloads).
digests = {'ASLCitizen2731_135-Test.hdf5': '5496c40077f9444a5fd9b9fc586cf16b7c3272c6e7b9c2d84166b186bd9603c6', 'ASLCitizen2731_135-Train.hdf5': '20e1e8dcf22b5f93348cd0d9fd9bbf38bb56d4810da2c160144546d24e31a921', 'ASLCitizen2731_135-Val.hdf5': '203934cda0ed05867f3d7a2475bd82c6f04f93abaf13d19875842dec352d6585'}
for filename, expected in digests.items():
    digest = hashlib.sha256()
    with (root / filename).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f'Incomplete or incorrect release file: {filename}')
groups = {}
report = {'label_policy': 'Case-insensitive exact text match; cross-dataset sign senses unreviewed.',
          'sources': ['ASL Citizen, Desai et al., NeurIPS 2023', 'WholeBodyPose, doi:10.1117/12.3096312'],
          'splits': {}}
used_names, used_signers = set(), set()
for role in ('train', 'val', 'test'):
    with (citizen / f'{role}.csv').open() as stream:
        rows = {r['Video file']: r for r in csv.DictReader(stream) if r['Gloss'].lower() in bank}
    signers = {r['Participant ID'] for r in rows.values()}
    if used_names.intersection(rows) or used_signers.intersection(signers):
        raise ValueError('Official split overlap')
    used_names.update(rows)
    used_signers.update(signers)
    features, labels, seen = [], [], set()
    with h5py.File(root / f'ASLCitizen2731_135-{role.title()}.hdf5') as archive:
        for key in archive:
            sample = archive[key]
            word = sample['label'][()].decode().lower()
            if word not in bank:
                continue
            name = Path(sample['video_name'][()].decode()).name
            if not name.lower().endswith('.mp4'):
                name += '.mp4'
            if name not in rows or rows[name]['Gloss'].lower() != word or name in seen:
                raise ValueError(f'Clip identity mismatch: {name}, {word}, {role}')
            seen.add(name)
            pose = sample['data'][()].transpose(0, 2, 1)[:, BODY + HAND, :]
            try:
                vector = describe(pose, float(sample['width'][()]) / float(sample['height'][()]))
            except ValueError:
                continue
            features.append(vector)
            labels.append(word)
    groups[role] = np.stack(features), np.asarray(labels)
    report['splits'][role] = dict(attempted=len(rows), matched_files=len(seen), usable=len(labels),
                                  signers=len(signers))
    print(role, report['splits'][role], flush=True)

train_x, train_y = groups['train']
val_x, val_y = groups['val']
best_score = -1
report['validation'] = {}
for kernel, alpha in [('linear', .1), ('linear', 1.), ('rbf', .1), ('rbf', 1.)]:
    layers = [StandardScaler(), Normalizer()]
    if kernel == 'rbf':
        layers.append(Nystroem(gamma=2., n_components=min(1200, len(train_y)), random_state=42))
    layers.append(RidgeClassifier(alpha=alpha, solver='cholesky'))
    model = make_pipeline(*layers)
    model.fit(train_x, train_y)
    score = int(sum(model.predict(val_x) == val_y))
    key = f'{kernel}/{alpha}'
    report['validation'][key] = score
    if score > best_score:
        best_score, best = score, model
        report['selected'] = key
test_x, test_y = groups['test']
pred = best.predict(test_x)
report['trained_words'] = len(best.classes_)
report['test_correct'] = int(sum(pred == test_y))
report['per_word'] = {}
for word in best.classes_:
    report['per_word'][word] = dict(correct=int(sum((pred == word) & (test_y == word))),
                                    attempted=sum(r['Gloss'].lower() == word for r in rows.values()))
joblib.dump(best, 'models/broad-pose-research.joblib', compress=3)
Path('reports/broad-pose.json').write_text(json.dumps(report, indent=2))
print(report['trained_words'], report['test_correct'], flush=True)
