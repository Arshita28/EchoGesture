"""Audit official WLASL metadata and local trimmed clips; never downloads or trains."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

SPLITS = ('train', 'validation', 'test')
FIELDS = ('video_id', 'label_id', 'label', 'split', 'official_split', 'signer_id',
          'source', 'url', 'fps', 'frame_start', 'frame_end', 'path', 'availability', 'sha256')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit(metadata, videos, class_count=30):
    """Select the official top-K order, preserving clip assignments and identities.

    videos must contain already trimmed clips named VIDEO_ID.mp4. File presence
    is not proof of decoding, correct signing, or permission to redistribute.
    """
    if class_count < 2:
        raise ValueError('Choose at least two classes')
    data = json.loads(Path(metadata).read_text())
    if not isinstance(data, list) or class_count > len(data):
        raise ValueError('Class count exceeds metadata vocabulary')
    rows, errors, warnings, labels = [], [], [], []
    ids, glosses, signers, hashes = {}, set(), defaultdict(set), defaultdict(list)
    for label_id, entry in enumerate(data[:class_count]):
        gloss = entry.get('gloss')
        if not isinstance(gloss, str) or not gloss.strip() or gloss in glosses:
            raise ValueError('Missing or duplicate gloss')
        glosses.add(gloss); labels.append(gloss)
        for item in entry.get('instances', []):
            video_id = str(item.get('video_id', ''))
            context = f'{gloss}/{video_id}'
            if not re.fullmatch(r'[0-9]+', video_id):
                errors.append(context + ': invalid video ID'); continue
            if video_id in ids:
                errors.append(context + ': duplicate video ID'); continue
            ids[video_id] = gloss
            official = item.get('split')
            split = 'validation' if official == 'val' else official
            if split not in SPLITS:
                errors.append(context + ': invalid split'); continue
            start, end, fps = (item.get(k) for k in ('frame_start', 'frame_end', 'fps'))
            if (type(start) is not int or start < 1 or type(end) is not int
                    or (end != -1 and end < start)
                    or not isinstance(fps, (int, float)) or isinstance(fps, bool)
                    or not 0 < fps < float('inf')):
                errors.append(context + ': invalid frame bounds or fps'); continue
            signer = item.get('signer_id')
            if signer is None or str(signer) in ('', '-1'):
                errors.append(context + ': unknown signer ID')
            else:
                signers[str(signer)].add(split)
            path = Path(videos).resolve() / (video_id + '.mp4')
            present = path.is_file() and path.stat().st_size > 0
            digest = sha256(path) if present else ''
            if digest:
                hashes[digest].append((video_id, split))
            rows.append(dict(video_id=video_id, label_id=label_id, label=gloss,
                split=split, official_split=official, signer_id=signer,
                source=item.get('source', ''), url=item.get('url', ''), fps=fps,
                frame_start=start, frame_end=end, path=str(path),
                availability='present_unverified' if present else 'missing', sha256=digest))
    counts = {label: {split: {'metadata': 0, 'present_unverified': 0} for split in SPLITS} for label in labels}
    for row in rows:
        count = counts[row['label']][row['split']]; count['metadata'] += 1
        count['present_unverified'] += row['availability'] == 'present_unverified'
    missing_partitions = [f'{label}/{split}' for label in labels for split in SPLITS
                          if counts[label][split]['present_unverified'] == 0]
    overlap = sum(len(parts) > 1 for parts in signers.values())
    if overlap:
        warnings.append(f'{overlap} signer IDs span official partitions; do not claim signer-disjoint evaluation.')
    duplicates = [group for group in hashes.values() if len(group) > 1]
    if duplicates:
        errors.append(f'{len(duplicates)} duplicate local file-content groups; investigate before extraction.')
    available = sum(r['availability'] == 'present_unverified' for r in rows)
    report = dict(dataset='WLASL v0.3', metadata_sha256=sha256(metadata),
        selection='First K glosses in official metadata order; no availability-based test selection',
        class_count=class_count, labels=labels, metadata_clips=len(rows),
        present_unverified_clips=available, missing_clips=len(rows)-available,
        split_counts=dict(Counter(r['split'] for r in rows)), counts=counts,
        signer_ids_crossing_official_partitions=overlap, duplicate_content_groups=len(duplicates),
        partitions_without_local_clips=missing_partitions, errors=errors, warnings=warnings,
        ready_for_decode_audit=bool(rows) and not errors and not missing_partitions,
        ready_for_training=False,
        note='Metadata and file audit only. No video decoding, URL reachability, sign correctness, '
             'license clearance, sufficient sample size, or recognition accuracy established. '
             'Official frame offsets use the metadata fps; local files must already be trimmed. '
             'Later extraction must not apply the same trim twice.')
    return report, rows


def write_outputs(report, rows, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'labels.json').write_text(json.dumps(report['labels'], indent=2) + '\n')
    with (output / 'manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata', type=Path)
    parser.add_argument('--videos', type=Path, default=Path('datasets/raw/wlasl/videos'))
    parser.add_argument('--classes', type=int, default=30)
    parser.add_argument('--output', type=Path, default=Path('datasets/preprocessed/wlasl'))
    args = parser.parse_args()
    try:
        report, rows = audit(args.metadata, args.videos, args.classes)
        write_outputs(report, rows, args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, str(exc) + '\n')
    print(json.dumps({k: report[k] for k in ('class_count', 'metadata_clips',
        'present_unverified_clips', 'missing_clips', 'split_counts',
        'signer_ids_crossing_official_partitions', 'errors', 'warnings',
        'ready_for_decode_audit', 'ready_for_training')}, indent=2))
    raise SystemExit(2 if report['errors'] else 0)
