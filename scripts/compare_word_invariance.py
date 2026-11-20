"""Compare handedness/timing candidates on original validation only."""
import hashlib,json
from pathlib import Path
import joblib,numpy as np
from asl_lab.word_invariant import InvariantWordMatcher
from scripts.train_word_background import load,metrics


def main():
    active=joblib.load('models/words-prototype.joblib');labels=active['labels']
    known=json.loads(Path('datasets/preprocessed/word-dense/extraction.json').read_text())['rows']
    negatives=json.loads(Path('datasets/preprocessed/word-negative-dense/extraction.json').read_text())['rows']
    rows=[{**r,'target':r['label'] if r['label'] in labels else 'NONE'} for r in known+negatives if r['status']=='extracted' and r['quality_pass'] and r['split'] in ['train','validation']]
    subsets={}
    for split in ['train','validation']:
        group=[r for r in rows if r['split']==split];subsets[split]=(np.array([load(r) for r in group]),np.array([r['target'] for r in group]))
    baseline=metrics(subsets['validation'][1],active['model'].predict(subsets['validation'][0]))
    results={};models={}
    for mirror,velocity in [(False,1.),(False,3.),(True,0.),(True,1.)]:
        name=f'mirror_{mirror}_velocity_{velocity}';model=InvariantWordMatcher(mirror,velocity).fit(*subsets['train'])
        result=metrics(subsets['validation'][1],model.predict(subsets['validation'][0]));models[name]=model;results[name]=result
        print(name,result,flush=True)
    eligible=[name for name,r in results.items() if r['known_correct']>=baseline['known_correct'] and r['other_rejected']>=baseline['other_rejected']]
    winner=max(eligible or list(results),key=lambda n:results[n]['balanced_utility'])
    path=Path('models/words-invariant-candidate.joblib');joblib.dump({**active,'name':'five_word_invariant_candidate_v6','model':models[winner]},path)
    report=dict(selected=winner,eligible=bool(eligible),baseline_validation=baseline,candidates=results,
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),production_replaced=False,
        note='Validation-only development; mirror transform is a hypothesis about dominant-hand variants, not proven universal linguistic invariance. No web fitting or selection.')
    Path('reports/word-invariant-comparison.json').write_text(json.dumps(report,indent=2)+'\n');print('Selected',winner,'eligible',bool(eligible))

if __name__=='__main__':main()
