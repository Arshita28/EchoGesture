import numpy as np
import pytest
from asl_lab.word_features import word_features, WORD_FEATURE_DIM


def sample():
    pose=np.zeros((33,4));pose[:,3]=1
    pose[11,:2]=[.7,.4];pose[12,:2]=[.3,.4]
    hand=np.array([[.4+i*.003,.5+i*.004,i*.001] for i in range(21)])
    return hand,hand+[.2,0,0],pose


def test_translation_scale_and_roll_invariance_preserve_body_location():
    left,right,pose=sample();expected=word_features(left,right,pose)
    assert expected.shape==(WORD_FEATURE_DIM,) and expected.dtype==np.float32
    np.testing.assert_equal(expected[-3:],[1,1,1])
    for angle in [0,.4]:
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        transformed=[]
        for a in [left,right,pose]:
            b=a.copy();b[:,:3]*=2;b[:,:2]=b[:,:2]@rotation.T+[.15,.1];transformed.append(b)
        np.testing.assert_allclose(word_features(*transformed),expected,atol=1e-6)
    moved=word_features(left+[.1,0,0],right,pose)
    np.testing.assert_allclose(moved[:126],expected[:126],atol=1e-6)
    assert moved[126]!=expected[126]  # Hand position is not normalized away.


def test_aspect_units_are_consistent():
    left,right,pose=sample();expected=word_features(left,right,pose)
    for a in [left,right,pose]:a[:,1]*=2
    np.testing.assert_allclose(word_features(left,right,pose,aspect=2),expected,atol=1e-6)


def test_missing_hand_and_body_are_masked_without_fake_coordinates():
    left,right,pose=sample();f=word_features(None,right,pose)
    assert np.count_nonzero(f[:63])==0
    np.testing.assert_equal(f[-3:],[0,1,1])
    assert not word_features(left,right,None).any()
    pose[11,3]=.1
    assert not word_features(left,right,pose).any()


def test_invalid_and_degenerate_landmarks():
    left,right,pose=sample()
    with pytest.raises(ValueError):word_features(left,right,pose,aspect=0)
    left[2,0]=np.nan
    with pytest.raises(ValueError):word_features(left,right,pose)
    f=word_features(np.zeros((21,3)),right,pose)
    assert f[-3]==0 and f[-2]==1


def test_uniform_sampling_includes_boundaries_and_quality_uses_explicit_masks():
    from scripts.extract_words import selected_indices, quality
    indices=selected_indices(100)
    assert indices[0]==0 and indices[-1]==99 and len(indices)==32
    assert (np.diff(indices)>=0).all()
    f=np.zeros((32,WORD_FEATURE_DIM),dtype=np.float32)
    assert not quality(f)['quality_pass']
    f[:,-1]=1;f[:16,-3]=1
    assert quality(f)['quality_pass'] and quality(f)['any_hand_coverage']==.5
    with pytest.raises(ValueError):selected_indices(7)


def test_sequence_gate_rejects_mislabeled_artifacts_and_duplicate_content(tmp_path):
    import json
    from asl_lab.word_features import WORD_FEATURE_VERSION
    from scripts.audit_word_sequences import audit
    x=np.zeros((32,WORD_FEATURE_DIM),np.float32);x[:,-3:]=1
    p=tmp_path/'sample.npz'
    np.savez_compressed(p,features=x,timestamps=np.arange(32)/25,
        feature_version=WORD_FEATURE_VERSION,video_sha256='fixture_hash')
    row=dict(status='extracted',video_id='1',label='book',label_id=0,
        split='train',feature_path=str(p),video_sha256='fixture_hash',signer_id=1)
    path=tmp_path/'extraction.json'
    path.write_text(json.dumps(dict(feature_version=WORD_FEATURE_VERSION,
        rows=[row,{**row,'video_id':'2','split':'test'}])))
    result,accepted=audit(path,['book','drink'])
    assert len(accepted)==1 and 'Duplicate video content' in result['errors'][0]
    assert not result['ready_for_training']
    row['label_id']=1
    path.write_text(json.dumps(dict(feature_version=WORD_FEATURE_VERSION,rows=[row])))
    result,_=audit(path,['book','drink'])
    assert 'Label ID mismatch' in result['errors'][0]


def test_cached_extraction_report_is_json_serializable(tmp_path,monkeypatch):
    import json,sys,hashlib
    import scripts.extract_words as module
    from asl_lab.word_features import WORD_FEATURE_VERSION
    output=tmp_path/'out';output.mkdir();videos=tmp_path/'videos';videos.mkdir()
    metadata=[]
    for i,label in enumerate(['book','drink']):
        video_id=str(i);p=videos/(video_id+'.mp4');p.write_bytes(video_id.encode())
        x=np.zeros((32,WORD_FEATURE_DIM),np.float32);x[:,-3:]=1
        np.savez_compressed(output/(video_id+'.npz'),features=x,timestamps=np.arange(32)/25,
            feature_version=WORD_FEATURE_VERSION,video_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),decode_info='{}')
        metadata.append(dict(gloss=label,instances=[dict(video_id=video_id,split='train',signer_id=i)]))
    path=tmp_path/'metadata.json';path.write_text(json.dumps(metadata))
    class FakeExtractor:
        def close(self):pass
    monkeypatch.setattr(module,'WordExtractor',FakeExtractor)
    monkeypatch.setattr(sys,'argv',['extract_words',str(path),'--classes','2','--videos',str(videos),'--output',str(output)])
    module.main()
    report=json.loads((output/'extraction.json').read_text())
    assert all(r['cached'] is True for r in report['rows'])
    assert report['quality_passes']==2
