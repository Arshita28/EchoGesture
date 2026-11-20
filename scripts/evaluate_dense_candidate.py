"""Regression-test dense candidate on already inspected web data, never fit it."""
import hashlib,json
from pathlib import Path
import joblib,numpy as np,cv2
from asl_lab.word_features import WordExtractor
from asl_lab.word_dense import crop_sequence,DENSE_VERSION
from asl_lab.word_model import classify_word
from scripts.evaluate_words_external import read_frames,stress_frames


def main():
    path=Path('models/words-dense-candidate.joblib');sha=hashlib.sha256(path.read_bytes()).hexdigest();bundle=joblib.load(path)
    sources=json.loads(Path('datasets/raw/web-evaluation/sources-segmented.json').read_text())
    output=Path('datasets/preprocessed/word-dense-regression');output.mkdir(parents=True,exist_ok=True)
    extractor=WordExtractor();rows=[]
    try:
        for source in sources:
            if source['status']!='downloaded':continue
            frames,fps=read_frames(source['path'])
            for condition in ['original','both_holds_1s']:
                cache=output/(source['video_id']+'-'+condition+'.npz')
                if cache.exists():
                    with np.load(cache,allow_pickle=False) as f:features=f['features']
                else:
                    changed=stress_frames(frames,fps,condition);all_features=[];times=[];next_time=0;extractor.model.reset()
                    for index,frame in enumerate(changed):
                        if index/fps+1e-7<next_time:continue
                        all_features.append(extractor.extract(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)));times.append(index/fps);next_time+=1/25
                    try:features,t,info=crop_sequence(all_features,times)
                    except ValueError:features=np.zeros((32,154),np.float32)
                    np.savez_compressed(cache,features=features,preprocessing=DENSE_VERSION)
                result=classify_word(bundle,features)
                rows.append(dict(video_id=source['video_id'],expected=source['label'],condition=condition,
                    predicted=result['label'],status=result['status'],correct=result['label']==source['label'],quality=result['quality']))
            print(source['label'],source['video_id'],'completed',flush=True)
    finally:extractor.close()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Model changed')
    summary={}
    for condition in ['original','both_holds_1s']:
        group=[r for r in rows if r['condition']==condition];summary[condition]=dict(correct=sum(r['correct'] for r in group),total=len(group),abstentions=sum(r['predicted'] is None for r in group))
    report=dict(model_sha256=sha,summary=summary,rows=rows,scope='Inspected web development regression only; not fresh validation or test. No fitting or tuning.')
    Path('reports/word-dense-regression.json').write_text(json.dumps(report,indent=2)+'\n');print(summary)

if __name__=='__main__':main()
