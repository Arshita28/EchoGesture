"""Bootstrap from source-labeled public clips, without inventing held-out signers."""
import json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,joblib,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from asl_lab.features import HandExtractor
from asl_lab.motion import motion_features,MOTION_VERSION
ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'data/processed/motion-public';output.mkdir(parents=True,exist_ok=True)
    extractor=HandExtractor();accepted=[];failures=[]
    paths=sorted((ROOT/'data/raw/signalpha/ASL_dynamic').rglob('*.avi'))
    try:
        for index,path in enumerate(paths,1):
            label=path.parent.name if path.parent.name in ('J','Z') else 'NONE'
            cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));frames=[];runs=[[]]
            if fps<=0:
                cap.release();failures.append({'path':str(path.relative_to(ROOT)),'reason':'invalid FPS'});continue
            stride=max(1,round(fps/10))
            for frame_index in range(n):
                ok,image=cap.read()
                if not ok:break
                if frame_index%stride:continue
                features,status=extractor.extract(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                if features is None:continue
                time=frame_index*1000/fps
                if runs[-1] and time-runs[-1][-1]['time']>750:runs.append([])
                runs[-1].append({'points':extractor.points,'time':time,'aspect':image.shape[1]/image.shape[0]})
            cap.release();frames=max(runs,key=len)
            if len(frames)>120:frames=[frames[i] for i in np.linspace(0,len(frames)-1,120).astype(int)]
            try:feature=motion_features(frames)
            except ValueError as error:
                failures.append({'path':str(path.relative_to(ROOT)),'reason':str(error)});continue
            row={'path':str(path.relative_to(ROOT)),'label':label,'source_label':path.parent.name,'frames':frames,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_group':'SignAlphaSet_unknown_signer_session','group_scope':'unknown; not independently partitioned'}
            accepted.append(row)
            print(f'{index}/{len(paths)} extracted: {path.parent.name}/{path.name} ({len(frames)} frames)',flush=True)
    finally:extractor.close()
    if set(r['label'] for r in accepted)!=set(['J','Z','NONE']):raise ValueError('Insufficient extracted classes; no model written')
    # Hyperparameters fixed before fitting. No validation/test split is invented.
    x=np.array([motion_features(r['frames']) for r in accepted]);y=np.array([r['label'] for r in accepted])
    model=make_pipeline(StandardScaler(),SVC(C=1,class_weight='balanced',probability=True,random_state=42));model.fit(x,y)
    bundle={'model':model,'version':MOTION_VERSION,'threshold':.7,'validation_status':'public_bootstrap_not_independently_validated','source':'SignAlphaSet v3, DOI 10.17632/8fmvr9m98w.3, CC BY 4.0','training_count':len(accepted)}
    joblib.dump(bundle,ROOT/'models/motion-public-bootstrap.joblib')
    (output/'sequences.json').write_text(json.dumps(accepted))
    report={'source':bundle['source'],'validation_status':bundle['validation_status'],'downloaded':len(paths),'extracted':len(accepted),'classes':{label:int(sum(r['label']==label for r in accepted)) for label in ['J','Z','NONE']},'failures':failures,'feature_version':MOTION_VERSION,'method':'Independent MediaPipe detection; 10fps sampling; longest contiguous detected segment; first-wrist trajectory normalization; 24-step resampling; balanced SVM C=1 fixed before training. All accepted source clips are training data.','evaluation':'No independent signer/session evaluation possible from supplied metadata. No accuracy estimate is reported. Source labels and sampled motion content need further expert verification; fresh webcam J/Z/background clips must be kept out of fitting for evaluation.'}
    (output/'bootstrap-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
