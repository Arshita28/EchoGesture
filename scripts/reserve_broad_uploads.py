"""Reserve one previously unavailable WLASL test recording per covered word."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

import joblib

root = Path('datasets/raw')
folder = root / 'broad-upload-holdout'
folder.mkdir(exist_ok=True)
dest = folder / 'manifest.json'
if dest.exists():
    raise SystemExit('Reuse existing frozen manifest')
known = set()
for base in (root, Path('mytest')):
    for path in base.rglob('*.mp4'):
        if folder not in path.parents:
            known.add(hashlib.sha256(path.read_bytes()).hexdigest())
labels = set(joblib.load('models/broad-cache-baseline.joblib').classes_)
rows = []
for entry in json.loads((root / 'wlasl/WLASL_v0.3.json').read_text()):
    if entry['gloss'] not in labels:
        continue
    for item in entry['instances']:
        if (item['split'] == 'test' and item['source'] in ('signschool','startasl','asldeafined','spreadthesign')
                and item['frame_start'] == 1 and item['frame_end'] == -1):
            rows.append(dict(word=entry['gloss'], id=item['video_id'], url=item['url'], provider=item['source']))
            break


def fetch(row):
    path = folder / (row['id'] + '.mp4')
    code = subprocess.run(['curl', '-fsSL', '--max-time', '20', '--max-filesize', '15000000',
                           row['url'], '-o', str(path)]).returncode
    if code == 0:
        row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    row['downloaded'] = code == 0


with ThreadPoolExecutor(4) as pool:
    list(pool.map(fetch, rows))
for row in rows:
    row['reserved'] = row['downloaded'] and row['sha256'] not in known
    if row['reserved']:
        known.add(row['sha256'])
dest.write_text(json.dumps(rows, indent=2))
print(sum(r['reserved'] for r in rows), 'reserved from', len(rows))
