"""Inventory WLASL label coverage against the official ASL Citizen splits."""
import csv
import json
from collections import defaultdict
from pathlib import Path

root = Path('datasets/raw/asl-citizen-selected')
metadata = json.loads(Path('datasets/raw/wlasl/WLASL_v0.3.json').read_text())
official = {}
for role in ('train', 'val', 'test'):
    by_word = defaultdict(list)
    with (root / f'{role}.csv').open() as stream:
        for row in csv.DictReader(stream):
            by_word[row['Gloss'].lower()].append(row)
    official[role] = by_word
rows = []
for entry in metadata:
    word = entry['gloss'].lower()
    rows.append(dict(
        word=word,
        wlasl_train=sum(i['split'] == 'train' for i in entry['instances']),
        wlasl_train_signer_ids=len({i['signer_id'] for i in entry['instances']
                                    if i['split'] == 'train' and i['signer_id'] >= 0}),
        citizen_official={s: len(group.get(word, [])) for s, group in official.items()},
        citizen_train_signers=len({r['Participant ID'] for r in official['train'].get(word, [])}),
        citizen_lex_codes=sorted({r['ASL-LEX Code'] for r in official['train'].get(word, [])}),
        citizen_cached_training_features=sum((root / r['Video file']).with_suffix('.npy').exists()
                                             for r in official['train'].get(word, []))))
Path('reports/broad-word-bank.json').write_text(json.dumps(rows, indent=2))
