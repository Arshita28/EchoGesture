import pytest
from scripts.acquire_wlasl import eligibility, inspect_video


def test_acquisition_rejects_unsupported_source_and_trim_policies():
    item=dict(url='https://example.invalid/video.mp4',frame_start=1,frame_end=-1)
    assert eligibility(item) is None
    assert eligibility({**item,'url':'http://example.invalid/video.mp4'})
    assert eligibility({**item,'url':'https://www.youtube.com/watch?v=x'})
    assert eligibility({**item,'url':'https://example.invalid/video.swf'})
    assert eligibility({**item,'frame_start':10})


def test_decode_audit_rejects_html_instead_of_video(tmp_path):
    p=tmp_path/'bad.mp4';p.write_bytes(b'<html>Not a video</html>')
    with pytest.raises(ValueError):inspect_video(p)
