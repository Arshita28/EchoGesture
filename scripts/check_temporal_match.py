import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,joblib
from asl_lab.motion import motion_features
from asl_lab.temporal_match import TemporalMatcher
from scripts.robust_motion import timing
root=Path(__file__).resolve().parents[1];roles=json.loads((root/'data/processed/motion-balanced-adaptation.json').read_text());public=json.loads((root/'data/processed/motion-public/sequences.json').read_text());local=[json.loads((root/'data/processed/motion'/f).read_text()) for f in roles['training_files']];train=public+local
m=TemporalMatcher().fit(np.array([motion_features(r['frames']) for r in train]),[r['label'] for r in train]);base=joblib.load(root/'models/motion.joblib');results=[]
for r in roles['reserved_results']:
 clip=json.loads((root/'data/processed/motion'/r['file']).read_text())
 for case,frames in [('original',clip['frames']),('start_pause',timing(clip['frames'],.2,0)),('end_pause',timing(clip['frames'],0,.2)),('both_pauses',timing(clip['frames'],.2,.2))]:
  scores=m.predict_proba(motion_features(frames).reshape(1,-1))[0];i=int(np.argmax(scores));label=str(m.classes_[i]);score=float(scores[i]);results.append({'expected':clip['label'],'case':case,'top':label,'score':score,'accepted':label if label!='NONE' and score>=.7 else None})
summary={case:{'correct':sum(r['accepted']==(None if r['expected']=='NONE' else r['expected']) for r in results if r['case']==case),'total':sum(r['case']==case for r in results)} for case in ['original','start_pause','end_pause','both_pauses']}
print(json.dumps(summary,indent=2));print(json.dumps([r for r in results if r['case']=='original'],indent=2))
joblib.dump({**base,'model':m,'validation_status':'temporal_matching_development_only'},root/'models/motion-aligned-candidate.joblib');(root/'data/processed/temporal-match-comparison.json').write_text(json.dumps({'note':'Training roles unchanged; development and synthetic timing stress only; similarity scores are heuristic. No independent accuracy claim.','summary':summary,'results':results},indent=2))
