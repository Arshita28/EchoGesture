"""Migrate v1 metadata only when every source image is square (v2 is identical)."""
import json
from pathlib import Path
import sys
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asl_lab.features import FEATURE_VERSION

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'data/processed/landmarks.jsonl'
records = [json.loads(line) for line in path.read_text().splitlines()]
for record in records:
    if record['feature_version'] not in ('wrist-max-radius-v1', FEATURE_VERSION):
        raise ValueError('Unsupported source feature version')
    with Image.open(ROOT / 'data' / record['path']) as image:
        if image.width != image.height:
            raise ValueError('Non-square source: re-extract with scripts/extract.py')
    record['feature_version'] = FEATURE_VERSION
path.write_text(''.join(json.dumps(record) + '\n' for record in records))
print(f'Verified square dimensions for {len(records)} samples; v2 geometry is exactly unchanged.')
