"""Preserve second-round results alongside the earlier screening evidence."""
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path

def main():
 path=Path('reports/showcase-retest19-results.json');r=json.loads(path.read_text());assert r['complete']
 previous=json.loads(Path('reports/showcase-fresh50-summary.json').read_text())
 assert r['model_sha256']==previous['model_sha256']==hashlib.sha256(Path(r['model']).read_bytes()).hexdigest()
 first=[x for n in ('10','40') for x in json.loads(Path('reports/showcase-fresh'+n+'-results.json').read_text())['rows']]
 assert not {x['sha256'] for x in first} & {x['sha256'] for x in r['rows']}
 assert not {x['url'] for x in first} & {x['url'] for x in r['rows']}
 groups=defaultdict(list)
 for row in r['rows']:groups[row['word']].append(row)
 table=[]
 for word in previous['promising']:
  rows=groups[word];n=len(rows);k=sum(x['correct'] for x in rows)
  table.append(dict(word=word,first_correct=2,first_total=2,retest_correct=k,retest_total=n,combined_correct=k+2,combined_total=n+2,remains_perfect_with_new_evidence=n>0 and k==n,misses=[dict(prediction=x['prediction'],status=x['result'].get('status'),http=x['http'],url=x['url']) for x in rows if not x['correct']]))
 summary=dict(model_sha256=r['model_sha256'],retest_correct=r['correct'],retest_total=r['attempted'],words_retested=sum(x['retest_total']>0 for x in table),per_word=table,still_perfect=[x['word'] for x in table if x['remains_perfect_with_new_evidence']],not_retested=[x['word'] for x in table if not x['retest_total']],limitations=['Retest selected words based on earlier inspected outcomes.','Exact-content/source exclusions applied; unfamiliar signers and reencoded duplicates cannot be guaranteed.','Only one or two additional dictionary clips per word; not proof of flawless live accuracy.','Model and runtime vocabulary unchanged.'])
 Path('reports/showcase-retest19-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with Path('reports/showcase-retest19-summary.csv').open('w') as f:
  writer=csv.writer(f);writer.writerow(['word','first_correct','first_total','retest_correct','retest_total','combined_correct','combined_total','wrong_predictions'])
  for x in table:writer.writerow([x[k] for k in ['word','first_correct','first_total','retest_correct','retest_total','combined_correct','combined_total']]+[', '.join(y['prediction'] or y['status'] or 'error' for y in x['misses'])])
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
