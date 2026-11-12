"""Audit local phrase clip metadata before extraction or training; does not train."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {'path', 'label', 'signer', 'session', 'split', 'source', 'usage_basis', 'asl_reviewed'}


def audit(manifest):
    manifest = Path(manifest).resolve()
    vocabulary = json.loads((ROOT / 'config/phrase_vocabulary.json').read_text())
    labels = {p['id'] for p in vocabulary['phrases']} | {'NONE'}
    counts = {split: {label: 0 for label in sorted(labels)} for split in ('train', 'validation', 'test')}
    errors, signers, hashes = [], {}, {}
    with manifest.open(newline='') as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            return {'ready': False, 'errors': ['Missing columns: ' + ', '.join(sorted(missing))], 'counts': counts}
        for number, row in enumerate(reader, 2):
            prefix = f'Row {number}: '
            if any(not row.get(key, '').strip() for key in REQUIRED):
                errors.append(prefix + 'All metadata fields must be filled'); continue
            split, label, signer = row['split'], row['label'], row['signer']
            if split not in counts or label not in labels:
                errors.append(prefix + 'Unknown partition or label'); continue
            if row['asl_reviewed'].strip().lower() != 'yes':
                errors.append(prefix + 'ASL content must be reviewed')
            if signer in signers and signers[signer] != split:
                errors.append(prefix + 'Signer crosses partitions')
            signers[signer] = split
            path = (manifest.parent / row['path']).resolve()
            if not path.is_file():
                errors.append(prefix + 'Clip missing'); continue
            digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in hashes:
                errors.append(prefix + 'Duplicate clip content (including across partitions)')
            hashes[digest] = split
            counts[split][label] += 1
    for split, totals in counts.items():
        absent = [label for label, count in totals.items() if count == 0]
        if absent:
            errors.append(f'{split}: missing classes: ' + ', '.join(absent))
    return {'ready': not errors, 'errors': errors, 'counts': counts,
            'note': 'Metadata gate only; does not verify fluency, license validity, video decoding or adequate sample size. No accuracy claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', help='Local CSV; paths relative to the CSV')
    args = parser.parse_args()
    report = audit(args.manifest)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['ready'] else 1)
