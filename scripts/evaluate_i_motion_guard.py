"""Measure held/repositioned-I rejection on synthetic trials from held-out poses."""
import json
from pathlib import Path
from app import create_app

client=create_app().test_client()
rows=[r for r in json.loads(Path('datasets/preprocessed/i-motion-negatives/sequences.json').read_text()) if r['role']=='test']
results=[]
for row in rows:
    response=client.post('/api/motion',json={'action':'predict','frames':row['frames']})
    assert response.status_code==200
    result=response.get_json()
    results.append(dict(path=row['path'],condition=row['condition'],predicted=result['label'],candidate=result['candidate'],status=result['status']))
previous=json.loads(Path('reports/i-motion-before.json').read_text())
report=dict(previous_wrong_j=sum(r['wrong_j'] for r in previous),wrong_j=sum(r['predicted']=='J' for r in results),
            displayed_j_candidates=sum(r['candidate']=='J' for r in results),total=len(results),rows=results,
            scope='Synthetic stress trials from20 held-out static I poses,8 movement conditions. These are correlated transformations, not independent video or live accuracy.')
Path('reports/i-motion-guard.json').write_text(json.dumps(report,indent=2))
assert report['wrong_j']==0 and report['displayed_j_candidates']==0
print({k:v for k,v in report.items() if k!='rows'})
