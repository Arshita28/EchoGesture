"""Dense extraction of existing other-word train/validation clips only.

Fresh dictionary test clips are never read by this preparation step.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from asl_lab.word_dense import extract_dense,DENSE_VERSION
from asl_lab.word_features import WordExtractor,WORD_FEATURE_VERSION
from scripts.extract_words import quality


def main():
    labels=json.loads(Path('config/word_prototype.json').read_text())['labels']
    source=json.loads(Path('datasets/preprocessed/words/quality-manifest.json').read_text())
    source=[r for r in source if r['label'] not in labels and r['split'] in ['train','validation']]
    output=Path('datasets/preprocessed/word-negative-dense');output.mkdir(parents=True,exist_ok=True)
    rows=[];extractor=WordExtractor()
    try:
        for item in source:
            path=Path('datasets/raw/wlasl/videos')/(item['video_id']+'.mp4');sha=hashlib.sha256(path.read_bytes()).hexdigest()
            cache=output/(item['video_id']+'.npz');row={**item,'feature_path':str(cache.resolve()),'video_sha256':sha,'negative_label':'NONE'}
            try:
                if cache.exists():
                    with np.load(cache,allow_pickle=False) as f:
                        if str(f['video_sha256'])!=sha or str(f['preprocessing'])!=DENSE_VERSION:raise ValueError('Stale negative cache')
                        x=f['features']
                else:
                    x,t,info=extract_dense(path,extractor)
                    np.savez_compressed(cache,features=x,timestamps=t,video_sha256=sha,feature_version=WORD_FEATURE_VERSION,preprocessing=DENSE_VERSION)
                row.update(status='extracted',**quality(x))
            except (ValueError,OSError) as exc:row.update(status='failed',quality_pass=False,error=str(exc))
            rows.append(row)
            (output/'extraction.json').write_text(json.dumps(dict(preprocessing=DENSE_VERSION,rows=rows),indent=2)+'\n')
            if len(rows)%10==0:print(len(rows),'of',len(source),'processed',flush=True)
    finally:extractor.close()
    print('Finished',len(rows),'clips;',sum(r['quality_pass'] for r in rows),'pass')

if __name__=='__main__':main()
