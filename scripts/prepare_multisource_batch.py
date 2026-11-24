"""Acquire WLASL clips for the five-word expansion with held-out providers per word."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from asl_lab.word_dense import extract_dense
from asl_lab.word_features import WordExtractor
from scripts.extract_words import quality

words = {'tomato', 'voice', 'grammar', 'impossible', 'vomit'}
providers = {'signschool', 'startasl', 'asldeafined', 'spreadthesign'}
root = Path('datasets/raw/expansion105')
root.mkdir(exist_ok=True)
rows = []
tracker = WordExtractor()
try:
    for entry in json.loads(Path('datasets/raw/wlasl/WLASL_v0.3.json').read_text()):
        if entry['gloss'] not in words:
            continue
        clips = [i for i in entry['instances'] if i['source'] in providers
                 and i['frame_start'] == 1 and i['frame_end'] == -1]
        held = {i['source'] for i in clips if i['split'] == 'test'}
        for clip in clips:
            if clip['split'] == 'train' and clip['source'] in held:
                continue
            if clip['split'] != 'train' and clip['source'] not in held:
                continue
            row = dict(**clip, word=entry['gloss'])
            rows.append(row)
            path = root / (clip['video_id'] + '.mp4')
            result = subprocess.run(['curl', '-fsSL', '--max-time', '20', '--retry', '1',
                                     '--max-filesize', '15000000', clip['url'], '-o', str(path)])
            row['available'] = result.returncode == 0
            if not row['available']:
                continue
            row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            if clip['split'] == 'train':
                try:
                    seq = extract_dense(path, tracker)[0]
                    row['usable'] = quality(seq)['quality_pass']
                    if row['usable']:
                        np.save(path.with_suffix('.npy'), seq)
                except ValueError as error:
                    row['error'] = str(error)
finally:
    tracker.close()
(root / 'manifest.json').write_text(json.dumps(rows, indent=2))
