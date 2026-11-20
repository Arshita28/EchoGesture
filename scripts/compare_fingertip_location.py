"""Fixed-weight fingertip-location experiment on the active word model's splits."""
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from asl_lab.temporal_match import aligned_distance
from asl_lab.word_features import WordExtractor
from asl_lab.word_dense import extract_dense
from scripts.train_word_background import metrics


def location(result, aspect):
    # Mouth center from pose mouth corners; upper chest approximated below shoulders.
    values = np.zeros((2, 5, 2))
    if not result.pose_landmarks:
        return values.ravel()
    pose = result.pose_landmarks.landmark
    def xy(p):
        return np.array([p.x, p.y / aspect])
    if min(pose[i].visibility for i in (11, 12)) < .5:
        return values.ravel()
    axis = xy(pose[11]) - xy(pose[12])
    scale = np.linalg.norm(axis)
    if scale < 1e-6:
        return values.ravel()
    chest = (xy(pose[11]) + xy(pose[12])) / 2 + .25 * np.array([-axis[1], axis[0]])
    mouth = (xy(pose[9]) + xy(pose[10])) / 2
    for side, hand in enumerate([result.left_hand_landmarks, result.right_hand_landmarks]):
        if hand:
            tips = np.array([xy(hand.landmark[i]) for i in (4, 8, 12, 16, 20)])
            values[side, :, 1] = np.linalg.norm(tips - chest, axis=1) / scale
            if min(pose[9].visibility, pose[10].visibility) >= .5:
                values[side, :, 0] = np.linalg.norm(tips - mouth, axis=1) / scale
    return values.ravel()


def track(path, tracker, times):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    targets = np.rint(times * fps).astype(int)
    frames = {}
    tracker.model.reset()
    next_time = 0
    try:
        for i in range(max(targets) + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Cannot read frame {i} of {path}')
            if i / fps + 1e-7 < next_time:
                continue
            next_time += 1/25
            h, w = frame.shape[:2]
            if max(h, w) > 640:
                frame = cv2.resize(frame, (round(w*640/max(h,w)), round(h*640/max(h,w))))
            result = tracker.model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if i in targets:
                frames[i] = location(result, frame.shape[1]/frame.shape[0])
    finally:
        cap.release()
    return np.stack([frames[i] for i in targets])


def main():
    raw = Path('datasets/raw')
    prep = Path('datasets/preprocessed')
    cache = prep/'word-fingertip-v1'
    cache.mkdir(exist_ok=True)
    bundle = joblib.load('models/words-prototype.joblib')
    groups = {'train': [], 'validation': [], 'regression': []}
    tracker = WordExtractor()
    def add(split, key, label, path, x, times):
        target = cache/(key + '.npy')
        if not target.exists():
            np.save(target, track(path, tracker, times))
        weights = np.ones(154)
        weights[126:130] = 2
        weights[130:144] = .5
        groups[split].append((key, label, np.column_stack([x*weights, np.load(target)]), x))
    try:
        for folder in ['word-dense','word-negative-dense']:
            rows = json.loads((prep/folder/'extraction.json').read_text())['rows']
            for r in rows:
                if r['split'] not in ('train','validation') or not r['quality_pass']:
                    continue
                key = r['video_id']
                path = next(p for f in ('wlasl','word-training-extension') if (p := raw/f/'videos'/(key+'.mp4')).exists())
                with np.load(r['feature_path']) as d:
                    add(r['split'], key, r['label'] if folder=='word-dense' else 'NONE', path, d['features'], d['timestamps'])
        sources = json.loads((raw/'fresh-dictionary/sources.json').read_text())['sources']
        for r in sources:
            path = Path(r['path'])
            x, times, _ = extract_dense(path, tracker)
            add('regression', r['video_id'], r['label'] if r['label'] in bundle['labels'] else 'NONE', path, x, times)
        path = raw/'cool-sense/ziggy-evaluation.mp4'
        x, times, _ = extract_dense(path, tracker)
        add('regression','ziggy','cool',path,x,times)
    finally:
        tracker.close()
    report = {}
    for split in ('validation','regression'):
        rows = []
        for key, label, x, old in groups[split]:
            nearest = min(groups['train'], key=lambda t: aligned_distance(x,t[2]))
            rows.append(dict(id=key,truth=label,active=str(bundle['model'].predict(old[None])[0]),candidate=nearest[1]))
        report[split] = dict(rows=rows, **{m: metrics([r['truth'] for r in rows],[r[m] for r in rows]) for m in ['active','candidate']})
    Path('reports/word-fingertip-location.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
