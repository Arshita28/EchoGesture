"""Import raw public J/Z recordings without unpickling downloaded label files.

Source: JordiEncaboBataller/sign-language-cv. The author advertises MIT terms
and describes the temporal dataset as one person. All imports are training-only.
"""
import hashlib
import json
import pickletools
from pathlib import Path
from collections import Counter

import numpy as np
from asl_lab.motion import motion_features


def read_labels(path):
    # The release uses an object array. Read its literal string list and memo
    # references as data, without executing pickle globals or constructors.
    with path.open('rb') as stream:
        np.lib.format.read_magic(stream)
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        if dtype.kind != 'O' or len(shape) != 1:
            raise ValueError('Unexpected label file schema')
        operations=pickletools.genops(stream.read())
        labels, memo, last, reading = [], {}, None, False
        for opcode, value, _ in operations:
            name=opcode.name
            if name=='MEMOIZE':
                memo[len(memo)]=last
                continue
            if name=='EMPTY_LIST':
                if reading:raise ValueError('Unexpected nested label list')
                reading=True
                last=None
                continue
            if name=='APPENDS' and reading:
                break
            if name in ('SHORT_BINUNICODE','BINUNICODE'):
                last=value
            elif name in ('BINGET','LONG_BINGET'):
                last=memo.get(value)
            else:
                last=None
            if reading and name in ('SHORT_BINUNICODE','BINUNICODE','BINGET','LONG_BINGET'):
                if last not in ('J','Z','STATIC'):
                    raise ValueError('Unexpected label in object array')
                labels.append(last)
        if len(labels)!=shape[0]:raise ValueError('Label count mismatch')
        return labels

root=Path('datasets/raw/public-jz-landmarks')
rows,seen=[],set()
report=dict(source='https://github.com/JordiEncaboBataller/sign-language-cv',
            terms='README advertises MIT; no standalone LICENSE file found',
            signer_scope='Author describes temporal training recordings as one person',
            geometry='Raw normalized MediaPipe coordinates. Aspect4/3 and FPS30 from capture source requested webcam configuration; actual captured dimensions/timestamps absent.',
            role='training only',files={})
for name,label_file in [('sequences.npy','labels.npy'),('static.npy','static_labels.npy')]:
    x=np.load(root/name,allow_pickle=False)
    labels=read_labels(root/label_file)
    if x.shape!=(len(labels),30,63) or not np.isfinite(x).all():raise ValueError('Invalid landmark sequences')
    report['files'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    report['files'][label_file]=hashlib.sha256((root/label_file).read_bytes()).hexdigest()
    for i,(sample,label) in enumerate(zip(x,labels)):
        sha=hashlib.sha256(sample.tobytes()).hexdigest()
        if sha in seen:continue
        seen.add(sha)
        frames=[dict(points=points.tolist(),time=float(j*1000/30),aspect=4/3)
                for j,points in enumerate(sample.reshape(30,21,3))]
        motion_features(frames)
        rows.append(dict(label=label if label!='STATIC' else 'NONE',frames=frames,
                         source_label=label,source='public_jz_single_signer',group='public_jz_p1',
                         path=f'{name}:{i}',role='train',sha256=sha,usable=True))
report['counts']=dict(Counter(r['label'] for r in rows))
output=Path('datasets/preprocessed/public-jz-landmarks');output.mkdir(exist_ok=True)
(output/'sequences.json').write_text(json.dumps(rows))
Path('reports/public-jz-import.json').write_text(json.dumps(report,indent=2))
print(report['counts'])
