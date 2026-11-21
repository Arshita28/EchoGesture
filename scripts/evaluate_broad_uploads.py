"""Score previously reserved external clips with the broad research model."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from asl_lab.broad_pose import describe_video

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepare-only', action='store_true')
parser.add_argument('--cached-baseline', action='store_true')
parser.add_argument('--reserved', action='store_true')
args = parser.parse_args()
stem = 'broad-cache-baseline' if args.cached_baseline else 'broad-pose-research'
root = Path('datasets/raw')
cache = Path('datasets/preprocessed') / ('broad-cached-uploads' if args.cached_baseline else 'broad-uploads')
cache.mkdir(parents=True, exist_ok=True)
sources = []
for row in json.loads((root / 'weak80-evaluation/sources.json').read_text()):
    sources.append((row['word'], Path(row['path'])))
for folder in ('expansion98', 'expansion100-eval', 'expansion105'):
    for row in json.loads((root / folder / 'manifest.json').read_text()):
        if not row['available']:
            continue
        if folder == 'expansion105' and row['split'] != 'test':
            continue
        if folder == 'expansion98' and row['role'] != 'eval':
            continue
        sources.append((row['word'], root / folder / (row['video_id'] + '.mp4')))
if args.reserved:
    folder = root / 'broad-upload-holdout'
    sources = [(row['word'], folder / (row['id'] + '.mp4'))
               for row in json.loads((folder / 'manifest.json').read_text()) if row['reserved']]
model = None if args.prepare_only else joblib.load(f'models/{stem}.joblib')
results, seen = [], set()
for word, path in sources:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest in seen:
        continue
    seen.add(digest)
    feature = cache / (digest + '.npy')
    result = dict(expected=word, sha256=digest, file=path.name)
    try:
        if not feature.exists():
            if args.cached_baseline:
                from asl_lab.broad_pose import from_cached
                from asl_lab.word_dense import extract_dense
                from asl_lab.word_features import WordExtractor
                tracker = WordExtractor()
                try:
                    vector = from_cached(extract_dense(path, tracker)[0])
                finally:
                    tracker.close()
            else:
                vector = describe_video(path)
            np.save(feature, vector)
        if model is not None:
            result['supported'] = word in model.classes_
            result['predicted'] = model.predict([np.load(feature)])[0]
    except ValueError as error:
        result['error'] = str(error)
    results.append(result)
if model is not None:
    suffix = 'reserved-uploads' if args.reserved else 'upload-regression'
    Path(f'reports/{stem}-{suffix}.json').write_text(json.dumps(results, indent=2))
    print(sum(r.get('predicted') == r['expected'] for r in results), '/', len(results))
else:
    print('Prepared', len(list(cache.glob('*.npy'))), 'feature files')
