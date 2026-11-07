"""Build a signer-disjoint manifest from the verified ASLYset archive layout."""
import csv
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asl_lab.features import LETTERS

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / 'data/raw/ASLYset/images'
    splits = {'User1': 'train', 'User2': 'train', 'User3': 'validation', 'User4': 'test'}
    records = []
    for path in sorted(directory.rglob('*.png')):
        group, label = path.parts[-3], path.parts[-2].upper()
        if label not in LETTERS:
            continue
        if group not in splits or not path.name.upper().startswith(f'{group}_{label}_'.upper()):
            raise ValueError(f'Unexpected dataset layout: {path}')
        records.append(dict(path=str(path.relative_to(ROOT / 'data')), label=label,
                            group='ASLYset_' + group, split=splits[group]))
    counts = Counter((r['group'], r['label']) for r in records)
    if len(counts) != 4 * 24 or any(n != 50 for n in counts.values()):
        raise ValueError(f'Expected 50 images per signer/static letter, found {len(records)} images across {len(counts)} groups/classes')
    manifest = ROOT / 'data/aslyset-manifest.csv'
    with manifest.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['path', 'label', 'group', 'split'])
        writer.writeheader()
        writer.writerows(records)
    print(f'Wrote {len(records)} samples to {manifest}')


if __name__ == '__main__':
    main()
