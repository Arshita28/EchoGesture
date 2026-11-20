"""Prepare a separate dense word experiment; existing sparse artifacts unchanged."""
import hashlib,json
from pathlib import Path
import numpy as np
from asl_lab.word_dense import extract_dense,DENSE_VERSION
from asl_lab.word_features import WordExtractor,WORD_FEATURE_VERSION
from scripts.extract_words import quality


def main():
    labels=json.loads(Path('config/word20.json').read_text())['labels']
    metadata=json.loads(Path('datasets/raw/wlasl/WLASL_v0.3.json').read_text())
    output=Path('datasets/preprocessed/word20-dense');output.mkdir(parents=True,exist_ok=True)
    entries=[]
    for e in metadata:
        if e['gloss'] not in labels:continue
        for i in e['instances']:
            for folder in ['datasets/raw/wlasl/videos','datasets/raw/word-training-extension/videos','datasets/raw/word20-extension/videos']:
                path=Path(folder)/(str(i['video_id'])+'.mp4')
                if path.exists():entries.append((e['gloss'],i,path));break
    extractor=WordExtractor();rows=[]
    try:
        for label,item,path in entries:
            video_id=str(item['video_id']);out=output/(video_id+'.npz');sha=hashlib.sha256(path.read_bytes()).hexdigest()
            row=dict(video_id=video_id,label=label,split='validation' if item['split']=='val' else item['split'],signer_id=item['signer_id'],video_sha256=sha,feature_path=str(out.resolve()))
            try:
                if out.exists():
                    with np.load(out,allow_pickle=False) as old:
                        if str(old['video_sha256'])!=sha or str(old['preprocessing'])!=DENSE_VERSION:raise ValueError('Stale dense cache')
                        x=old['features'];info=json.loads(str(old['crop_info']))
                else:
                    x,t,info=extract_dense(path,extractor)
                    np.savez_compressed(out,features=x,timestamps=t,video_sha256=sha,feature_version=WORD_FEATURE_VERSION,preprocessing=DENSE_VERSION,crop_info=json.dumps(info))
                row.update(status='extracted',**quality(x),**info)
            except (ValueError,OSError) as exc:row.update(status='failed',quality_pass=False,error=str(exc))
            rows.append(row)
            (output/'extraction.json').write_text(json.dumps(dict(preprocessing=DENSE_VERSION,rows=rows),indent=2)+'\n')
            print(label,video_id,row['status'],row['quality_pass'],flush=True)
    finally:extractor.close()

if __name__=='__main__':main()
