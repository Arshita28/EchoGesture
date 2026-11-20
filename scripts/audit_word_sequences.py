"""Gate extracted word sequences before training; never changes official splits."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from asl_lab.word_features import WORD_FEATURE_VERSION, WORD_FEATURE_DIM
from scripts.extract_words import SEQUENCE_FRAMES, quality

SPLITS=('train','validation','test')


def audit(extraction, labels):
    report=json.loads(Path(extraction).read_text())
    if report['feature_version']!=WORD_FEATURE_VERSION:raise ValueError('Extraction feature version mismatch')
    if not labels or len(set(labels))!=len(labels):raise ValueError('Missing or duplicate labels')
    counts={label:{split:0 for split in SPLITS} for label in labels}
    errors=[];accepted=[];hashes={};signers=defaultdict(set);seen_ids=set()
    for row in report['rows']:
        if row['status']!='extracted':continue
        video_id=row['video_id']
        if video_id in seen_ids:
            errors.append('Duplicate video ID: '+video_id);continue
        seen_ids.add(video_id)
        try:
            if row['label'] not in counts or row['split'] not in SPLITS:raise ValueError('Invalid label or split')
            if row['label_id']!=labels.index(row['label']):raise ValueError('Label ID mismatch')
            with np.load(row['feature_path'],allow_pickle=False) as sample:
                x=sample['features'];times=sample['timestamps']
                if str(sample['feature_version'])!=WORD_FEATURE_VERSION:raise ValueError('Feature version mismatch')
                if str(sample['video_sha256'])!=row['video_sha256']:raise ValueError('Video hash mismatch')
                if x.shape!=(SEQUENCE_FRAMES,WORD_FEATURE_DIM) or not np.isfinite(x).all():raise ValueError('Invalid feature array')
                if times.shape!=(SEQUENCE_FRAMES,) or not np.isfinite(times).all() or (np.diff(times)<0).any() or times[-1]<=times[0]:raise ValueError('Invalid timestamps')
                if not np.isin(x[:,-3:],[0,1]).all():raise ValueError('Invalid presence masks')
                if not quality(x)['quality_pass']:continue
            digest=row['video_sha256']
            if digest in hashes:raise ValueError('Duplicate video content with '+hashes[digest])
            hashes[digest]=video_id
            counts[row['label']][row['split']]+=1
            signers[str(row['signer_id'])].add(row['split'])
            accepted.append(row)
        except (ValueError,OSError,KeyError) as exc:errors.append(video_id+': '+str(exc))
    missing={label:[split for split in SPLITS if not values[split]] for label,values in counts.items()}
    missing={label:parts for label,parts in missing.items() if parts}
    return dict(feature_version=WORD_FEATURE_VERSION,labels=labels,counts=counts,
        accepted_sequences=len(accepted),labels_with_all_partitions=[label for label in labels if label not in missing],
        missing_partitions=missing,errors=errors,
        signer_ids_crossing_partitions=sum(len(v)>1 for v in signers.values()),
        all_classes_have_partition_coverage=not missing and not errors,
        ready_for_training=False,
        note='Partition coverage and feature-integrity checks only. Vocabulary requires explicit declaration, '
             'sample sufficiency review and background strategy before training. Do not auto-drop classes '
             'or move validation/test clips into training. Official splits are not signer-disjoint.'),accepted


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction',type=Path,default=Path('datasets/preprocessed/words/extraction.json'))
    parser.add_argument('--labels',type=Path,default=Path('datasets/preprocessed/wlasl/labels.json'))
    parser.add_argument('--output',type=Path,default=Path('datasets/preprocessed/words'))
    args=parser.parse_args()
    result,accepted=audit(args.extraction,json.loads(args.labels.read_text()))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'quality-manifest.json').write_text(json.dumps(accepted,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(2 if result['errors'] else 0)
