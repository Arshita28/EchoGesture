"""Benchmark the published Microsoft ASL Citizen I3D checkpoint on reserved clips.

Requires the authors' ASL-citizen-code checkout, their checkpoints_v1 I3D ZIP,
PyTorch and OpenCV. This is a pretrained comparison, not an EchoGesture fit.
"""
import csv
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

root = Path('datasets/raw/broad-pose')
sys.path.insert(0, str((root / 'ms-baselines/I3D').resolve()))
from aslcitizen_dataset import load_rgb_frames_from_video
from pytorch_i3d import InceptionI3d

with Path('datasets/raw/asl-citizen-selected/train.csv').open() as stream:
    words = sorted({r['Gloss'].strip() for r in csv.DictReader(stream)})
bank = {r['gloss'].lower() for r in json.loads(Path('datasets/raw/wlasl/WLASL_v0.3.json').read_text())}
selected = [i for i, word in enumerate(words) if word.lower() in bank]
assert len(words) == 2731 and len(selected) == 1159
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = InceptionI3d(400, in_channels=3)
model.replace_logits(len(words))
with zipfile.ZipFile(root / 'i3d.zip') as z:
    weights, = [n for n in z.namelist() if n.endswith(('.pt', '.pth')) and '__MACOSX/' not in n]
    with z.open(weights) as stream:
        model.load_state_dict(torch.load(stream, weights_only=True, map_location='cpu'))
model.to(device).eval()
folder = Path('datasets/raw/broad-upload-holdout')
results = []
with torch.inference_mode():
    for row in json.loads((folder / 'manifest.json').read_text()):
        if not row['reserved']:
            continue
        started = time.perf_counter()
        frames = load_rgb_frames_from_video(str(folder / (row['id'] + '.mp4')))
        if not len(frames):
            raise ValueError(f"Decode failed: {row['id']}")
        # Deterministic end padding; retain the reference decoder's BGR ordering.
        frames = np.pad(frames, ((0, 64-len(frames)), (0, 0), (0, 0), (0, 0)), mode='edge')
        h, w = frames.shape[1:3]
        top, left = round((h-224)/2), round((w-224)/2)
        frames = frames[:, top:top+224, left:left+224]
        tensor = torch.from_numpy(frames.transpose(3, 0, 1, 2)).unsqueeze(0).to(device)
        logits = F.interpolate(model(tensor), 64, mode='linear', align_corners=False).amax(dim=2)[0]
        ranks = logits.cpu().numpy()[selected].argsort()[::-1][:5]
        top5 = [words[selected[i]].lower() for i in ranks]
        results.append(dict(expected=row['word'], id=row['id'], top5=top5,
                            video_ms=round((time.perf_counter() - started) * 1000, 1)))
report = dict(vocabulary=len(selected), device=str(device),
              video_p95_ms=float(np.percentile([r['video_ms'] for r in results], 95)), source='Microsoft ASL-citizen-code checkpoints_v1, I3D',
              correct=sum(r['expected'] == r['top5'][0] for r in results),
              top5_correct=sum(r['expected'] in r['top5'] for r in results), total=len(results), rows=results)
Path('reports/broad-rgb-reserved-uploads.json').write_text(json.dumps(report, indent=2))
print('RGB reference:', report['correct'], '/',len(results),flush=True)
