"""Compare wrist weighting on the three inspected cool clips and dictionary who controls."""
import json
from pathlib import Path

import joblib
import numpy as np

from asl_lab.word_dense import extract_dense
from asl_lab.word_features import WordExtractor
from asl_lab.temporal_match import aligned_distance


def main():
    model = joblib.load('models/words-prototype.joblib')['model']
    sources = json.loads(Path('datasets/raw/fresh-dictionary/sources.json').read_text())['sources']
    clips = {}
    for source in sources:
        if source['label'] in ('who', 'cool'):
            key = source['video_id']
            with np.load(Path('datasets/preprocessed/fresh-dictionary', key + '-original.npz')) as data:
                clips[key] = (source['label'], data['features'])
    extractor = WordExtractor()
    try:
        clips['ziggy'] = ('cool', extract_dense('datasets/raw/cool-sense/ziggy-evaluation.mp4', extractor)[0])
    finally:
        extractor.close()
    report = {}
    for key, (label, x) in clips.items():
        slot = int(x[:, 151:153].sum(axis=0).argmax())
        visible = x[:, 151 + slot] > .5
        wrist = x[:, 126 + 2*slot:128 + 2*slot]
        nose_visible = visible & (x[:, 144] >= .5)
        result = dict(label=label, most_visible_hand=['left','right'][slot],
                      median_wrist_y=float(np.median(wrist[visible, 1])),
                      median_wrist_to_nose=float(np.median(np.linalg.norm(
                          wrist[nose_visible] - x[nose_visible, 130:132], axis=1))),
                      predictions={})
        for weight in (0, 2, 4):
            weights = np.ones(x.shape[1])
            weights[126:130] = weight
            weights[130:144] = .5
            distances = [aligned_distance(x * weights, t * weights) for t in model.templates_]
            result['predictions'][weight] = str(model.labels_[np.argmin(distances)])
        assert result['predictions'][2] == model.predict(x[None])[0]
        report[key] = result
    Path('reports/word-wrist-diagnostic.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
