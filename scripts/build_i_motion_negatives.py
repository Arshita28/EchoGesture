"""Create labelled I stress trials from distinct official static training/evaluation poses.

These are synthetic sequences, not independent camera recordings. Training poses
and held-out poses retain their original partition and never cross roles.
"""
import json
from pathlib import Path
import numpy as np

rows=[json.loads(line) for line in Path('datasets/preprocessed/landmarks.jsonl').read_text().splitlines() if line.strip()]
selected={}
for role,split in [('train','train'),('test','test')]:
    subset=[r for r in rows if r['label']=='I' and r['split']==split]
    selected[role]=subset[::max(1,len(subset)//20)][:20]
training_groups={r['group'] for r in selected['train']}
assert not training_groups & {r['group'] for r in selected['test']}
output=[]
for role,poses in selected.items():
    for index,row in enumerate(poses):
        shape=np.array(row['features']).reshape(21,3)*.2+[.5,.5,0]
        t=np.linspace(0,1,30)
        paths={'held':np.zeros((30,2)),
               'jitter':np.column_stack([.006*np.sin(t*20),.006*np.cos(t*17)]),
               'horizontal':np.column_stack([.12*np.minimum(t*2,1),t*0]),
               'vertical':np.column_stack([t*0,.12*np.minimum(t*2,1)]),
               'upward':np.column_stack([t*0,-.12*np.minimum(t*2,1)]),
               'leftward':np.column_stack([-.12*np.minimum(t*2,1),t*0]),
               'up_diagonal':np.column_stack([-.10*np.minimum(t*2,1),-.12*np.minimum(t*2,1)]),
               'diagonal':np.column_stack([.10*np.minimum(t*2,1),.12*np.minimum(t*2,1)])}
        for condition,path in paths.items():
            points=np.tile(shape[None],(30,1,1));points[:,:,:2]+=path[:,None]
            frames=[dict(points=p.tolist(),time=float(j*1000/30),aspect=1) for j,p in enumerate(points)]
            output.append(dict(label='NONE',source_label='I',role=role,frames=frames,usable=True,
                               source='synthetic_i_stress',group=row['group'],
                               path=f"I-{role}-{index}-{condition}",condition=condition))
folder=Path('datasets/preprocessed/i-motion-negatives');folder.mkdir(exist_ok=True)
(folder/'sequences.json').write_text(json.dumps(output))
print({role:sum(r['role']==role for r in output) for role in selected})
