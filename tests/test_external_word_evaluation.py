import numpy as np
from scripts.evaluate_words_external import stress_frames,summary


def test_stress_cases_preserve_source_and_timing_holds():
    frames=[np.full((30,40,3),100+i,dtype=np.uint8) for i in range(10)]
    dim=stress_frames(frames,10,'dim_55_percent')
    assert dim[0].mean()==55 and frames[0].mean()==100
    for condition,length in [('leading_hold_1s',20),('trailing_hold_1s',20),('both_holds_1s',30)]:
        changed=stress_frames(frames,10,condition)
        assert len(changed)==length
    assert stress_frames(frames,10,'low_resolution_160')[0].shape==frames[0].shape


def test_summary_counts_rejections_and_failures_in_all_attempt_accuracy():
    rows=[dict(status='prototype_match',correct=True),dict(status='insufficient_detection',correct=False),dict(status='failed',correct=False)]
    result=summary(rows)
    assert result['all_attempt_accuracy']==1/3
    assert result['accuracy_among_matches']==1
    assert result['detection_abstentions']==1 and result['failures']==1
