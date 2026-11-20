import json
import pytest
from scripts.audit_wlasl import audit, write_outputs


def metadata(tmp_path):
    rows=[]
    for label_id, gloss in enumerate(['book', 'drink']):
        instances=[]
        for index, split in enumerate(['train', 'val', 'test']):
            instances.append(dict(video_id=str(label_id*10+index), split=split,
                signer_id=index, frame_start=1, frame_end=-1, fps=25,
                url='https://example.invalid/clip', source='fixture'))
        rows.append(dict(gloss=gloss, instances=instances))
    path=tmp_path/'metadata.json';path.write_text(json.dumps(rows));return path


def test_preserves_official_splits_and_does_not_claim_missing_data_ready(tmp_path):
    path=metadata(tmp_path)
    report, rows=audit(path,tmp_path/'videos',2)
    assert report['missing_clips']==6
    assert report['split_counts']=={'train':2,'validation':2,'test':2}
    assert not report['ready_for_training'] and not report['ready_for_decode_audit']
    assert rows[1]['official_split']=='val' and rows[1]['split']=='validation'
    write_outputs(report,rows,tmp_path/'out')
    assert json.loads((tmp_path/'out/labels.json').read_text())==['book','drink']


def test_duplicate_content_and_signer_leakage_are_reported(tmp_path):
    path=metadata(tmp_path);data=json.loads(path.read_text())
    data[0]['instances'][1]['signer_id']=0;path.write_text(json.dumps(data))
    videos=tmp_path/'videos';videos.mkdir()
    (videos/'0.mp4').write_bytes(b'duplicate');(videos/'1.mp4').write_bytes(b'duplicate')
    report,_=audit(path,videos,2)
    assert report['signer_ids_crossing_official_partitions']==1
    assert report['duplicate_content_groups']==1 and report['errors']
    assert not report['ready_for_decode_audit']


def test_invalid_ids_bounds_and_duplicate_ids_cannot_enter_manifest(tmp_path):
    path=metadata(tmp_path);data=json.loads(path.read_text())
    data[0]['instances'][0]['video_id']='../escape'
    data[0]['instances'][1]['frame_end']=0
    data[1]['instances'][0]['video_id']='2';path.write_text(json.dumps(data))
    report,rows=audit(path,tmp_path,2)
    assert len(report['errors'])==3
    assert len(rows)==3
    with pytest.raises(ValueError):audit(path,tmp_path,3)


def test_present_files_still_need_decode_and_feature_validation(tmp_path):
    path=metadata(tmp_path);videos=tmp_path/'videos';videos.mkdir()
    for i in [0,1,2,10,11,12]:(videos/f'{i}.mp4').write_bytes(str(i).encode())
    report,_=audit(path,videos,2)
    assert report['ready_for_decode_audit']
    assert not report['ready_for_training']
