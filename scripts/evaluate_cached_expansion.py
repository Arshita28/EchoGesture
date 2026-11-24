"""Train a vocabulary expansion on cached official ASL Citizen training clips."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone

from asl_lab.word_svm_features import summarize, summarize_batch
from scripts.extract_words import quality

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--include-weak-extension', action='store_true')
parser.add_argument('--expand-to-100', action='store_true')
parser.add_argument('--expand-to-105', action='store_true')
parser.add_argument('--repair-ok-his', action='store_true')
parser.add_argument('--grammar-only', action='store_true')
parser.add_argument('--four-word-batch', action='store_true')
args = parser.parse_args()
if args.four_word_batch and any((args.grammar_only, args.expand_to_100, args.expand_to_105, args.include_weak_extension, args.repair_ok_his)):
    parser.error('--four-word-batch runs a separate experiment')
if args.grammar_only and any((args.expand_to_100, args.expand_to_105, args.include_weak_extension, args.repair_ok_his)):
    parser.error('--grammar-only runs the isolated 81-word experiment')
if args.repair_ok_his and not args.expand_to_105:
    parser.error('--repair-ok-his requires --expand-to-105')
batch = {'tomato', 'voice', 'grammar', 'impossible', 'vomit'}
root = Path('datasets/raw/asl-citizen-selected')
base = joblib.load('models/word80-priority-wlasl-citizen-svm.joblib')
old = set(base.classes_)
bank = {row['gloss'] for row in json.loads(Path('datasets/raw/wlasl/WLASL_v0.3.json').read_text())}
rows = {}
for split in ('train', 'val', 'test'):
    with (root / f'{split}.csv').open() as stream:
        rows[split] = list(csv.DictReader(stream))
for key in ('Participant ID', 'Video file'):
    sets = [set(r[key] for r in group) for group in rows.values()]
    assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i))

def cached(row):
    return (root / row['Video file']).with_suffix('.npy')

counts = Counter(r['Gloss'].lower() for r in rows['train'] if cached(r).exists())
added = set('address bell brag chop close expect gift head his hope octopus ok other page price road scream tongue'.split())
if args.expand_to_100 or args.expand_to_105:
    added.update(('jewelry', 'squeeze'))
if args.expand_to_105:
    added.update(batch)
if args.grammar_only:
    added = {'grammar'}
if args.four_word_batch:
    added = {'grammar', 'scream', 'other', 'address', 'close'}
if not added <= bank - old or any(counts[w] < 15 for w in added):
    raise ValueError('Expansion training clips or WLASL labels missing')
with np.load('/tmp/echo80-neural-data.npz') as data:
    assert list(base.classes_) == list(data['labels'])
    x = list(summarize_batch(data['train_x']))
    y = list(data['labels'][data['train_y']])
for row in rows['train']:
    word = row['Gloss'].lower()
    if word in added and cached(row).exists():
        seq = np.load(cached(row))
        if quality(seq)['quality_pass']:
            x.append(summarize(seq))
            y.append(word)
if args.include_weak_extension:
    audit = json.loads(Path('reports/expansion98-upload-audit.json').read_text())
    for clip in audit['clips']:
        if clip['role'] == 'train' and clip.get('usable'):
            path = Path('datasets/raw/expansion98') / (clip['id'] + '.npy')
            x.append(summarize(np.load(path)))
            y.append(clip['word'])
if args.expand_to_105 or args.grammar_only or args.four_word_batch:
    folder = Path('datasets/raw/expansion105')
    manifest = json.loads((folder / 'manifest.json').read_text())
    external_hashes = {r['sha256'] for r in manifest if r['available'] and r['split'] != 'train'}
    trained_hashes = set()
    for clip in manifest:
        if clip['word'] not in added or clip['split'] != 'train' or not clip.get('usable'):
            continue
        digest = clip['sha256']
        if digest in external_hashes:
            raise ValueError('WLASL training/evaluation overlap')
        if digest not in trained_hashes:
            x.append(summarize(np.load(folder / (clip['video_id'] + '.npy'))))
            y.append(clip['word'])
            trained_hashes.add(digest)
if args.repair_ok_his:
    folder = Path('datasets/raw/expansion105-repairs')
    prior = json.loads(Path('reports/expansion98-upload-audit.json').read_text())
    reserved = {r['sha256'] for r in prior['clips'] if r['role'] == 'eval'}
    for clip in json.loads((folder / 'manifest.json').read_text()):
        if clip['split'] == 'train' and clip.get('usable'):
            digest = clip['sha256']
            if digest in reserved | external_hashes:
                raise ValueError('Repair clip matches reserved evaluation data')
            if digest not in trained_hashes:
                x.append(summarize(np.load(folder / (clip['video_id'] + '.npy'))))
                y.append(clip['word'])
                trained_hashes.add(digest)
candidate = clone(base).fit(x, y)
report = {'new_training_counts': {w: y.count(w) for w in sorted(added)}}
for split in ('val', 'test'):
    scores = {'old': [0, 0, 0], 'new': [0, 0]}
    per_word = {}
    for row in rows[split]:
        word = row['Gloss'].lower()
        if word not in old | added:
            continue
        group = scores['old' if word in old else 'new']
        group[-1] += 1
        tally = per_word.setdefault(word, [0, 0])
        tally[1] += 1
        if cached(row).exists():
            seq = np.load(cached(row))
            if quality(seq)['quality_pass']:
                features = [summarize(seq)]
                correct = int(candidate.predict(features)[0] == word)
                tally[0] += correct
                group[-2] += correct
                if word in old:
                    group[0] += int(base.predict(features)[0] == word)
    report[split] = {'scores': scores, 'per_word': per_word}
    print(split, scores, flush=True)
stem = 'word98-weak-extension' if args.include_weak_extension else 'word-cached-expansion'
if args.expand_to_100:
    stem = 'word100-weak-extension' if args.include_weak_extension else 'word100-expansion'
if args.expand_to_105:
    stem = 'word105-weak-extension' if args.include_weak_extension else 'word105-expansion'
if args.repair_ok_his:
    stem += '-repairs'
if args.grammar_only:
    stem = 'word81-grammar'
if args.four_word_batch:
    stem = 'word85-four-additions'
joblib.dump(candidate, f'models/{stem}-candidate.joblib')
Path(f'reports/{stem}.json').write_text(json.dumps(report, indent=2))
