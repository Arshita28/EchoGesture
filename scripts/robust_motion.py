"""Training-only timing augmentation; reserved checks are development, not independent accuracy."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from asl_lab.motion import motion_features
ROOT=Path(__file__).resolve().parents[1]

def timing(frames,lead,trail):
    times=np.array([f['time'] for f in frames]);t=(times-times[0])/(times[-1]-times[0]);p=np.array([f['points'] for f in frames]).reshape(len(frames),63)
    grid=np.linspace(0,1,24);source=np.clip((grid-lead)/(1-lead-trail),0,1)
    points=np.stack([np.interp(source,t,p[:,i]) for i in range(63)],axis=1).reshape(24,21,3)
    return [{'points':v.tolist(),'time':float(i*2500/23),'aspect':frames[0]['aspect']} for i,v in enumerate(points)]

def main():
    roles=json.loads((ROOT/'data/processed/motion-balanced-adaptation.json').read_text())
    public=json.loads((ROOT/'data/processed/motion-public/sequences.json').read_text())
    local=[json.loads((ROOT/'data/processed/motion'/f).read_text()) for f in roles['training_files']]
    held=[{**json.loads((ROOT/'data/processed/motion'/r['file']).read_text()),'file':r['file']} for r in roles['reserved_results']]
    train=public+local;x=[];y=[];weights=[]
    variations=[(0,0),(.15,0),(0,.15),(.25,.15),(.15,.25),(.3,.3)]
    for index,r in enumerate(train):
        for lead,trail in variations:
            x.append(motion_features(timing(r['frames'],lead,trail)));y.append(r['label']);weights.append(3 if index>=len(public) else 1)
    model=make_pipeline(StandardScaler(),SVC(C=1,class_weight='balanced',probability=True,random_state=42))
    model.fit(np.array(x),np.array(y),svc__sample_weight=np.array(weights))
    base=joblib.load(ROOT/'models/motion.joblib');results=[]
    for r in held:
        for case,frames in [('original',r['frames']),('start_pause',timing(r['frames'],.2,0)),('end_pause',timing(r['frames'],0,.2)),('both_pauses',timing(r['frames'],.2,.2))]:
            result={'expected':r['label'],'case':case}
            for name,m in [('before',base['model']),('candidate',model)]:
                scores=m.predict_proba(motion_features(frames).reshape(1,-1))[0];i=int(np.argmax(scores));label=str(m.classes_[i]);score=float(scores[i]);result[name]={'top':label,'score':score,'accepted':label if label!='NONE' and score>=.7 else None}
            results.append(result)
    summary={}
    for name in ['before','candidate']:
        summary[name]={}
        for case in ['original','start_pause','end_pause','both_pauses']:
            subset=[r for r in results if r['case']==case];summary[name][case]={'correct':sum(r[name]['accepted']==(None if r['expected']=='NONE' else r['expected']) for r in subset),'total':len(subset)}
    joblib.dump({**base,'model':model,'validation_status':'timing_augmented_local_model_not_independently_validated'},ROOT/'models/motion-timing-candidate.joblib')
    report={'note':'No new user data fitted. Only pre-existing approved training clips receive temporal start/end-pause augmentation. Reserved original clips and their derived timing stresses stay out of fitting. Stress tests are synthetic perturbations of already-inspected development clips, not independent accuracy. Cutoff unchanged at 70%.','summary':summary,'results':results}
    (ROOT/'data/processed/motion-timing-comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
