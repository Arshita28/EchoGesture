"""Research video encoder using the published ASL Citizen I3D reference."""
import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


class BroadRGB:
    def __init__(self, root):
        root = Path(root)
        sys.path.insert(0, str((root / 'broad-pose/ms-baselines/I3D').resolve()))
        from aslcitizen_dataset import load_rgb_frames_from_video
        from pytorch_i3d import InceptionI3d
        self.read = load_rgb_frames_from_video
        with (root / 'asl-citizen-selected/train.csv').open() as stream:
            words = sorted({r['Gloss'].strip() for r in csv.DictReader(stream)})
        bank = {r['gloss'].lower() for r in json.loads((root / 'wlasl/WLASL_v0.3.json').read_text())}
        self.indices = [i for i, word in enumerate(words) if word.lower() in bank]
        self.labels = np.array([words[i].lower() for i in self.indices])
        if len(words) != 2731 or len(self.labels) != 1159:
            raise ValueError('Unexpected reference label mapping')
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = InceptionI3d(400, in_channels=3)
        self.model.replace_logits(len(words))
        with zipfile.ZipFile(root / 'broad-pose/i3d.zip') as z:
            weights, = [n for n in z.namelist() if n.endswith(('.pt', '.pth')) and '__MACOSX/' not in n]
            with z.open(weights) as stream:
                self.model.load_state_dict(torch.load(stream, weights_only=True, map_location='cpu'))
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def encode(self, path):
        frames = self.read(str(path))
        if not len(frames):
            raise ValueError(f'No decoded frames: {path.name}')
        frames = np.pad(frames, ((0, 64-len(frames)), (0, 0), (0, 0), (0, 0)), mode='edge')
        h, w = frames.shape[1:3]
        top, left = round((h-224)/2), round((w-224)/2)
        tensor = torch.from_numpy(frames[:, top:top+224, left:left+224].transpose(3, 0, 1, 2))
        pooled = self.model.extract_features(tensor.unsqueeze(0).to(self.device))
        logits = self.model.logits(pooled).squeeze(-1).squeeze(-1)
        scores = F.interpolate(logits, 64, mode='linear', align_corners=False).amax(dim=2)[0]
        embedding = pooled.mean(dim=(2, 3, 4))[0].cpu().numpy()
        return embedding, scores.cpu().numpy()[self.indices]
