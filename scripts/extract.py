"""Manifest input preserves human-verified group identities and split assignments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageOps
from asl_lab.features import HandExtractor, FEATURE_VERSION, LETTERS


def extract(manifest, output):
    manifest, output = Path(manifest), Path(output)
    rows = list(csv.DictReader(manifest.open()))
    seen_groups, seen_hashes = {}, {}
    prepared = []
    for row in rows:
        if row['label'] not in LETTERS:
            raise ValueError('Only static A–Y excluding J is supported')
        if not row['group'].strip() or row['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Verified group and split required')
        path = (manifest.parent / row['path']).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for key, seen in ((row['group'], seen_groups), (digest, seen_hashes)):
            if key in seen and seen[key] != row['split']:
                raise ValueError('Group or identical image crosses splits')
            seen[key] = row['split']
        prepared.append((row, path, digest))
    started = time.perf_counter()
    extractor, records, failures = HandExtractor(), [], []
    try:
        for index, (row, path, digest) in enumerate(prepared, 1):
            if index % 500 == 0:
                print(f'Processing {index}/{len(prepared)}', flush=True)
            try:
                with Image.open(path) as image:
                    rgb = np.array(ImageOps.exif_transpose(image).convert('RGB'))
                features, status = extractor.extract(rgb)
                if features is None:
                    failures.append({'path': row['path'], 'label': row['label'], 'group': row['group'], 'status': status})
                    continue
                records.append({**row, 'sha256': digest, 'features': features.tolist(), 'feature_version': FEATURE_VERSION})
            except (OSError, ValueError) as exc:
                failures.append({'path': row['path'], 'label': row['label'], 'group': row['group'], 'status': str(exc)})
    finally:
        extractor.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(r) + '\n' for r in records))
    summary = {'input': len(rows), 'extracted': len(records), 'seconds': time.perf_counter() - started,
               'success_by_group': dict(Counter(r['group'] for r in records)),
               'success_by_class': dict(Counter(r['label'] for r in records)),
               'failure_by_group': dict(Counter(r['group'] for r in failures)),
               'failure_by_class': dict(Counter(r['label'] for r in failures)), 'failures': failures}
    output.with_suffix('.extraction.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'failures'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', help='CSV: path,label,group,split; paths relative to CSV')
    parser.add_argument('--output', default='data/processed/landmarks.jsonl')
    args = parser.parse_args()
    extract(args.manifest, args.output)
