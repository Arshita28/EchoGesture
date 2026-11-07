"""Benchmark real local server processing with public images and an empty frame."""
import io
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app

ROOT = Path(__file__).resolve().parents[1]


def main():
    client = create_app().test_client()
    blank = io.BytesIO()
    Image.new('RGB', (640, 480)).save(blank, 'JPEG')
    cases = [('blank', blank.getvalue())]
    for label in ['a', 'b', 'c', 'l', 'y']:
        path = ROOT / f'data/raw/ASLYset/images/User4/{label}/User4_{label.upper()}_1.png'
        cases.append((label.upper(), path.read_bytes()))
    results = []
    for name, content in cases:
        timings, response = [], None
        for index in range(12):
            started = time.perf_counter()
            response = client.post('/api/predict', data={'image': (io.BytesIO(content), 'frame.png')})
            if response.status_code != 200:
                raise RuntimeError(response.get_data(as_text=True))
            if index >= 2:
                timings.append((time.perf_counter() - started) * 1000)
        result = response.json
        result.pop('landmarks', None)
        results.append({'input': name, 'median_ms': float(np.median(timings)),
                        'p95_ms': float(np.percentile(timings, 95)), 'response': result})
    report = {'platform': platform.platform(), 'architecture': platform.machine(),
              'python': platform.python_version(), 'measurement': 'Flask test client end-to-end server request; excludes browser encoding, network and 180 ms scheduling delay',
              'warmup_per_case': 2, 'iterations_per_case': 10, 'cases': results}
    (ROOT / 'reports/runtime.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
