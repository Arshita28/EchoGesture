"""Read-only validation of the local raw/preprocessed dataset inventory."""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'datasets'

def verify():
    errors = []
    with (ROOT / 'inventory/files.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected = {row['path'] for row in rows}
    for row in rows:
        path = ROOT / row['path']
        if not path.is_file():
            errors.append(row['path'] + ': missing'); continue
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        if path.stat().st_size != int(row['bytes']) or digest.hexdigest() != row['sha256']:
            errors.append(row['path'] + ': changed')
    actual = {p.relative_to(ROOT).as_posix() for folder in ('raw', 'preprocessed')
              for p in (ROOT / folder).rglob('*') if p.is_file()}
    print(f'Checked {len(rows)} inventoried files; {len(errors)} missing/changed; {len(actual - expected)} new files.')
    for error in errors:
        print(error)
    return not errors

if __name__ == '__main__':
    raise SystemExit(0 if verify() else 1)
