"""Exercise the 105-word model on reserved videos using the multipart upload route."""
import json
import os
from pathlib import Path

from flask import Flask
from asl_lab.word_routes import word_routes

folder = Path('datasets/raw/expansion105')
manifest = json.loads((folder / 'manifest.json').read_text())
clips = [(r['word'], folder / (r['video_id'] + '.mp4'), 'new_word')
         for r in manifest if r['available'] and r['split'] == 'test']
prior = json.loads(Path('reports/expansion98-upload-audit.json').read_text())
for row in prior['clips']:
    if row['role'] == 'eval' and row['word'] in ('his', 'ok'):
        path = Path('datasets/raw/expansion98') / (row['id'] + '.mp4')
        clips.append((row['word'], path, 'previously_inspected'))

results = []
for variant in ('word105-svm', 'word105-repairs-svm'):
    os.environ['ECHOGESTURE_WORD_MODEL'] = variant
    app = Flask(__name__)
    app.register_blueprint(word_routes(Path.cwd()))
    client = app.test_client()
    assert len(client.get('/api/words/status').get_json()['words']) == 105
    for word, path, role in clips:
        with path.open('rb') as stream:
            response = client.post('/api/words/predict', data={'video': (stream, path.name)})
        result = response.get_json() or {}
        results.append(dict(model=variant, word=word, id=path.stem, role=role,
                            http=response.status_code, predicted=result.get('label')))
Path('reports/word105-external-uploads.json').write_text(json.dumps(results, indent=2))
