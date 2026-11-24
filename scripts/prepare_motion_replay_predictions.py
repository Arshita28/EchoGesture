"""Include the static classifier outputs used by Alphabet Studio in replay."""
import json
from pathlib import Path
import joblib
import numpy as np
from asl_lab.features import normalize

root=Path('datasets/preprocessed')
rows=[r for r in json.loads((root/'motion-extension/sequences.json').read_text()) if r['role']=='test' and r['usable']]
roles=json.loads((root/'motion-balanced-adaptation.json').read_text())
rows += [dict(**json.loads((root/'motion'/r['file']).read_text()),path=r['file'],role='test',usable=True,source='local_development') for r in roles['reserved_results']]
model=joblib.load('models/classifier.joblib')['model']
for row in rows:
    frames=row['frames']
    x=np.stack([normalize(f['points'],f['aspect']) for f in frames])
    probabilities=model.predict_proba(x)
    for frame,scores in zip(frames,probabilities):
        index=int(np.argmax(scores));score=float(scores[index])
        frame['static_result']=dict(status='ok' if score>=.3 else 'uncertain',
                                    label=str(model.classes_[index]) if score>=.3 else None,
                                    candidate=str(model.classes_[index]),confidence=score)
(root/'motion-extension/replay-with-static.json').write_text(json.dumps(rows))
