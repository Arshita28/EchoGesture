"""Summarize first-look dictionary evaluation without modifying the model."""
import csv,json
from collections import defaultdict
from pathlib import Path

def main():
 reports=[json.loads(Path('reports/showcase-fresh'+n+'-results.json').read_text()) for n in ('10','40')]
 assert all(r['complete'] for r in reports)
 assert len({r['model_sha256'] for r in reports})==1
 rows=[x for r in reports for x in r['rows']]
 assert len({r['sha256'] for r in rows})==len(rows)
 ranking=json.loads(Path('reports/showcase-word-ranking.json').read_text())
 byword=defaultdict(list)
 for row in rows:byword[row['word']].append(row)
 table=[]
 for word in ranking['selected']:
  group=byword[word];n=len(group);k=sum(r['correct'] for r in group)
  category='promising_2_of_2' if n>=2 and k==n else 'limited_1_of_1' if n==1 and k==1 else 'mixed' if k else 'failed' if n else 'not_tested'
  table.append(dict(word=word,correct=k,total=n,category=category,misses=[r['prediction'] or r['result'].get('status') or 'error' for r in group if not r['correct']]))
 out=dict(model_sha256=reports[0]['model_sha256'],correct=sum(r['correct'] for r in rows),total=len(rows),words_with_coverage=sum(bool(v) for v in byword.values()),per_word=table,promising=[r['word'] for r in table if r['category']=='promising_2_of_2'],limitations=['Two dictionary videos per word are screening evidence, not reliable general accuracy estimates.','Source and exact-content overlap excluded; signer/reencoded overlap unverified.','These clips are now inspected evaluation regressions and must never be fitted.','Original unchanged 81-class model, not a retrained or restricted showcase model.'])
 Path('reports/showcase-fresh50-summary.json').write_text(json.dumps(out,indent=2)+'\n')
 with Path('reports/showcase-fresh50-summary.csv').open('w') as f:
  writer=csv.writer(f);writer.writerow(['word','correct','attempted','assessment','wrong_predictions'])
  for r in table:writer.writerow([r['word'],r['correct'],r['total'],r['category'],', '.join(r['misses'])])
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
