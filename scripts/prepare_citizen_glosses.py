"""Cache ASL Citizen clips and dense features for explicitly selected glosses."""
import argparse
import csv
import json
import struct
import subprocess
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from asl_lab.word_dense import extract_dense
from asl_lab.word_features import WordExtractor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('words', nargs='+')
words = {w.upper() for w in parser.parse_args().words}
root = Path('datasets/raw/asl-citizen-selected')
entries = json.loads((root / 'entries.json').read_text())
url = 'https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip'


def fetch(name):
    path = root / name
    start, size, length, crc, method = entries[name]
    if path.exists() and zlib.crc32(path.read_bytes()) == crc:
        return path
    chunk = subprocess.check_output(['curl', '-fsSL', '--retry', '2', '--max-time', '45',
                                    '--range', f'{start}-{start + size + 2047}', url + f'?offset={start}'])
    name_size, extra_size = struct.unpack_from('<HH', chunk, 26)
    offset = 30 + name_size + extra_size
    payload = chunk[offset:offset + size]
    data = zlib.decompress(payload, -15) if method == 8 else payload
    if len(data) != length or zlib.crc32(data) != crc:
        raise ValueError(f'Invalid ZIP entry for {name}')
    path.write_bytes(data)
    return path


names = []
for split in ('train', 'val', 'test'):
    with (root / f'{split}.csv').open() as stream:
        names.extend(r['Video file'] for r in csv.DictReader(stream) if r['Gloss'] in words)
tracker = WordExtractor()
try:
    with ThreadPoolExecutor(6) as pool:
        for path in pool.map(fetch, names):
            cache = path.with_suffix('.npy')
            if cache.exists():
                continue
            try:
                seq = extract_dense(path, tracker)[0]
            except ValueError as error:
                print(path.name, error)
                continue
            np.save(cache, seq)
finally:
    tracker.close()
