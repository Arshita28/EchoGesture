import io
import json
import numpy as np
import pytest
from PIL import Image
from asl_lab.features import normalize, FEATURE_VERSION
from asl_lab.predict import Stabilizer
from asl_lab.train import load_records, train
from app import create_app


def test_translation_and_scale_invariance():
    points = np.random.default_rng(2).normal(size=(21, 3))
    np.testing.assert_allclose(normalize(points), normalize(points * 3 + [10, 20, 30]), atol=1e-12)
    with pytest.raises(ValueError):
        normalize(np.zeros((21, 3)))


def test_same_hand_geometry_across_frame_aspects():
    square_points = np.random.default_rng(7).uniform(.1, .8, size=(21, 3))
    rectangular_points = square_points.copy()
    rectangular_points[:, 1] *= 16 / 9
    np.testing.assert_allclose(normalize(square_points),
        normalize(rectangular_points, image_aspect=16 / 9), atol=1e-12)
    assert not np.allclose(normalize(square_points), normalize(rectangular_points))


def test_stability_clears_stale_letter():
    state = Stabilizer(3)
    assert [state.update('A') for _ in range(3)] == [None, None, 'A']
    assert state.update(None) is None
    assert state.update('A') is None


def synthetic_records():
    # Infrastructure test only: these are NOT sign landmarks or accuracy evidence.
    records = []
    rng = np.random.default_rng(42)
    for split in ('train', 'validation', 'test'):
        for label, offset in [('A', -1), ('B', 1)]:
            for i in range(6):
                records.append(dict(features=(rng.normal(0, .1, 63) + offset).tolist(),
                    label=label, group=split, split=split, sha256=f'{split}-{label}-{i}', feature_version=FEATURE_VERSION))
    return records


def test_training_and_leakage_guard(tmp_path):
    rows = synthetic_records()
    path = tmp_path / 'samples.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    result = train(path, tmp_path / 'models', tmp_path / 'reports')
    assert set(result['models']) == {'svm', 'random_forest', 'knn'}
    assert (tmp_path / 'models/classifier.joblib').exists()
    rows[-1]['group'] = 'train'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match='crosses'):
        load_records(path)


def test_api_invalid_inputs_and_page():
    client = create_app().test_client()
    assert client.get('/').status_code == 200
    assert client.get('/api/status').status_code == 200
    assert client.post('/api/predict').status_code == 400
    assert client.post('/api/predict', data={'image': (io.BytesIO(b'bad'), 'bad.jpg')}).status_code == 400


def test_no_hand_api(monkeypatch):
    class Extractor:
        def extract(self, image):
            return None, 'no_hand'
    monkeypatch.setattr('app.HandExtractor', Extractor)
    image = io.BytesIO()
    Image.new('RGB', (30, 30)).save(image, 'PNG')
    image.seek(0)
    response = create_app().test_client().post('/api/predict', data={'image': (image, 'blank.png')})
    assert response.json['status'] == 'no_hand'
    assert response.json['label'] is None


def test_video_and_images_use_consistent_static_detection(monkeypatch):
    modes = []
    class Extractor:
        def __init__(self, static_image_mode=True):
            modes.append(static_image_mode)
        def extract(self, image):
            return None, 'no_hand'
        def close(self):
            pass
    monkeypatch.setattr('app.HandExtractor', Extractor)
    client = create_app().test_client()
    def request_frame(**fields):
        image = io.BytesIO()
        Image.new('RGB', (30, 30)).save(image, 'PNG')
        image.seek(0)
        return client.post('/api/predict', data={'image': (image, 'blank.png'), **fields})
    assert request_frame(mode='video', stream='one').status_code == 200
    assert request_frame(mode='video', stream='one').status_code == 200
    assert request_frame(mode='video', stream='two').status_code == 200
    assert request_frame().status_code == 200
    assert modes == [True]


def test_cutoff_changes_acceptance_without_changing_score():
    from asl_lab.predict import classify
    class Model:
        classes_ = np.array(['A', 'B'])
        def predict_proba(self, features):
            return np.array([[.45, .55]])
    baseline = classify({'model': Model(), 'threshold': .70}, np.zeros(63))
    live = classify({'model': Model(), 'threshold': .30}, np.zeros(63))
    assert baseline['status'] == 'uncertain' and baseline['label'] is None
    assert live['status'] == 'ok' and live['label'] == 'B'
    assert baseline['confidence'] == live['confidence'] == .55


def test_geometry_features_are_finite_and_capture_spread():
    from asl_lab.geometry import geometry_features
    points = np.random.default_rng(3).normal(size=(2, 63))
    result = geometry_features(points)
    assert result.shape == (2, 288)
    assert np.isfinite(result).all()
    changed = points.copy().reshape(2,21,3)
    changed[0,8] += [1,0,0]
    assert not np.allclose(geometry_features(changed.reshape(2,63))[0,63:],result[0,63:])
