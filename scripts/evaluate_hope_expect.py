"""Compare a two-word handshape specialist with the expansion on official splits."""
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from asl_lab.word_invariant import mirrored
from asl_lab.word_svm_features import summarize
from scripts.extract_words import quality

root = Path('datasets/raw/asl-citizen-selected')
pair = {'hope', 'expect'}


def handshape(sequence):
    if sequence[:, 151].sum() > sequence[:, 152].sum():
        sequence = mirrored(sequence)
    features = []
    for block in np.array_split(sequence, 4):
        for slot in range(2):
            visible = block[block[:, 151 + slot] > 0]
            hands = visible[:, slot * 63:(slot + 1) * 63]
            features.append(hands.mean(axis=0) if len(hands) else np.zeros(63))
    return np.concatenate(features)


def samples(split):
    with (root / f'{split}.csv').open() as stream:
        for row in csv.DictReader(stream):
            word = row['Gloss'].lower()
            path = (root / row['Video file']).with_suffix('.npy')
            yield word, np.load(path) if path.exists() else None


train = [(word, handshape(seq)) for word, seq in samples('train')
         if word in pair and seq is not None and quality(seq)['quality_pass']]
for row in json.loads(Path('reports/expansion98-upload-audit.json').read_text())['clips']:
    if row['role'] == 'train' and row['word'] in pair and row.get('usable'):
        seq = np.load(Path('datasets/raw/expansion98') / (row['id'] + '.npy'))
        train.append((row['word'], handshape(seq)))
labels, features = zip(*train)
specialist = make_pipeline(StandardScaler(), SVC(C=1)).fit(features, labels)
base = joblib.load('models/word100-weak-extension-candidate.joblib')
report = {}
for split in ('val', 'test'):
    scores = {}
    for word, seq in samples(split):
        if word not in base.classes_:
            continue
        counts = scores.setdefault(word, [0, 0, 0])
        counts[2] += 1
        if seq is None or not quality(seq)['quality_pass']:
            continue
        original = base.predict([summarize(seq)])[0]
        revised = specialist.predict([handshape(seq)])[0] if original in pair else original
        counts[0] += int(original == word)
        counts[1] += int(revised == word)
    report[split] = scores
Path('reports/word100-hope-expect.json').write_text(json.dumps(report, indent=2))
