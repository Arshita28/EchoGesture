"""Local-only analysis of consented debug photos; never train on these photos."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asl_lab.features import HandExtractor, normalize
from asl_lab.train import load_records, make_candidates
from asl_lab.predict import classify

ROOT = Path(__file__).resolve().parents[1]


def main():
    training = [r for r in load_records(ROOT / 'data/processed/landmarks.jsonl') if r['split'] == 'train']
    x, y = np.array([r['features'] for r in training]), np.array([r['label'] for r in training])
    models = make_candidates(len(training))
    for model in models.values():
        model.fit(x, y)
    rows = list(csv.DictReader((ROOT / 'data/manifest.csv').open()))
    results = []
    reference = HandExtractor()
    ref_image = np.array(Image.open(ROOT / 'data/raw/ASLYset/images/User4/b/User4_B_1.png').convert('RGB'))
    reference.extract(ref_image)
    reference_handedness = reference.handedness
    reference.close()
    for static in (True, False):
        extractor = HandExtractor(static_image_mode=static)
        try:
            for row in rows:
                image = np.array(Image.open(ROOT / 'data' / row['path']).convert('RGB'))
                repeats = []
                for _ in range(8):
                    features, status = extractor.extract(image)
                    result = {'detector_status': status, 'handedness': extractor.handedness}
                    if features is not None:
                        result['models'] = {name: classify({'model': model, 'threshold': .70}, features)
                                            for name, model in models.items()}
                        mirrored = features.reshape(21, 3).copy()
                        mirrored[:, 0] *= -1
                        result['svm_mirrored_diagnostic_only'] = classify({'model': models['svm'], 'threshold': .70}, mirrored.reshape(63))
                    repeats.append(result)
                results.append({'file': Path(row['path']).name, 'label': row['label'],
                    'mode': 'static' if static else 'tracking', 'repeats': repeats})
        finally:
            extractor.close()
    report = {'note': 'Diagnostic photos excluded from fitting. All models fitted only on original public training signers. Repeated photos are not an independent video or accuracy study. MediaPipe handedness assumes mirrored input; labels are detector metadata, not a verified physical-hand identity.',
              'public_B_reference_handedness': reference_handedness, 'cases': results}
    output = ROOT / 'data/processed/webcam-model-diagnostics.json'
    output.write_text(json.dumps(report, indent=2))
    print('Public reference handedness:', reference_handedness)
    for case in results:
        last = case['repeats'][-1]
        scores = {name: (r['candidate'], round(r['confidence'], 3)) for name, r in last.get('models', {}).items()}
        print(case['mode'], case['file'][:8], last['detector_status'], last['handedness'], scores,
              'mirrored svm', last.get('svm_mirrored_diagnostic_only', {}).get('confidence'))
    print('Local-only report:', output)


if __name__ == '__main__':
    main()
