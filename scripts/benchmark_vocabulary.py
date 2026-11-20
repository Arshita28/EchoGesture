"""Local closed-vocabulary research benchmark with disjoint metadata signer IDs."""
import json
from time import perf_counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from asl_lab.word_model import WordTemplateClassifier
from scripts.train_word_background import load


def pooled(x):
    return np.stack([np.concatenate([block.mean(axis=0) for block in
                    np.split(seq.reshape(32, 154), 4)]) for seq in x])


def main():
    rows = []
    for folder in ('word-dense', 'word-negative-dense'):
        manifest = Path('datasets/preprocessed', folder, 'extraction.json')
        rows.extend(r for r in json.loads(manifest.read_text())['rows'] if r['quality_pass'])
    if len({r['video_sha256'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source hash')
    sources = json.loads(Path('datasets/raw/web-evaluation/sources-segmented.json').read_text())
    reserved = {s['url'] for s in sources}
    dictionary = json.loads(Path('datasets/raw/fresh-dictionary/sources.json').read_text())['sources']
    from scripts.acquire_word_training import dictionary_mirror_ids
    metadata = json.loads(Path('datasets/raw/wlasl/WLASL_v0.3.json').read_text())
    excluded = dictionary_mirror_ids(metadata, dictionary)
    excluded.update(str(i['video_id']) for e in metadata for i in e['instances'] if i['url'] in reserved)
    rows = [r for r in rows if r['video_id'] not in excluded]
    groups = {}
    for r in rows:
        groups.setdefault(r['label'], []).append(r['signer_id'])
    counts = {w: dict(clips=len(ids), signers=len(set(ids))) for w, ids in groups.items()}
    ranked = sorted(groups, key=lambda w: (-len(set(groups[w])), -len(groups[w]), w))
    signers = sorted({r['signer_id'] for r in rows if r['signer_id'] > 0})
    rng = np.random.default_rng(42)
    best = ()
    # Choose roles using only signer/label coverage before looking at predictions.
    for _ in range(5000):
        p = rng.permutation(signers)
        dev, test = set(p[:5]), set(p[5:10])
        train = set(signers) - dev - test
        eligible = [w for w in ranked if all(set(groups[w]) & g for g in (dev, test))
                    and len(set(groups[w]) & train) >= 2][:20]
        score = (len(eligible), sum(len(set(groups[w]) & train) for w in eligible))
        if score > best:
            best, words, roles = score, eligible, (train, dev, test)
    datasets = []
    inventory = {}
    for role, ids in zip(('train','development','test'), roles):
        part = [r for r in rows if r['label'] in words and r['signer_id'] in ids]
        datasets.append((np.stack([load(r) for r in part]), np.array([r['label'] for r in part])))
        inventory[role] = [r['video_id'] for r in part]
    (x, y), (dx, dy), (tx, ty) = datasets
    dtw = WordTemplateClassifier().fit(x, y)
    scaler = StandardScaler().fit(pooled(x))
    linear = LogisticRegression(C=.1, class_weight='balanced', max_iter=1000).fit(scaler.transform(pooled(x)), y)
    # Fold scaler into linear weights, quantize per output class.
    weights = linear.coef_ / scaler.scale_
    bias = linear.intercept_ - weights @ scaler.mean_
    scale = np.maximum(abs(weights).max(axis=1), 1e-10) / 127
    q = np.rint(weights / scale[:,None]).astype(np.int8)
    path = Path('models/word-benchmark-linear.npz')
    np.savez_compressed(path, weights=q, scales=scale, bias=bias, labels=linear.classes_)
    def predict_linear(a):
        return linear.classes_[(pooled(a) @ q.T * scale + bias).argmax(axis=1)]
    predictors = {'templates':dtw.predict, 'quantized_linear_weights':predict_linear}
    development = {k:int((f(dx)==dy).sum()) for k,f in predictors.items()}
    selected = max(development, key=development.get)
    pred = predictors[selected](tx)
    template_path = Path('models/word-benchmark-templates-int8.npz')
    scales = np.maximum(np.abs(dtw.templates_).max(axis=(0, 1)), 1e-10) / 127
    np.savez_compressed(template_path, templates=np.rint(dtw.templates_/scales).astype(np.int8),
                        scales=scales, labels=y)
    with np.load(template_path) as artifact:
        restored = WordTemplateClassifier().fit(artifact['templates'] * artifact['scales'], artifact['labels'])
    quantized_predictions = []
    timings = []
    for example in tx:
        start = perf_counter()
        quantized_predictions.append(restored.predict(example[None])[0])
        timings.append((perf_counter()-start)*1000)
    report = dict(quantized_template_test_correct=int((ty==quantized_predictions).sum()),
                  template_artifact_bytes=template_path.stat().st_size,
                  desktop_classifier_only_p95_ms=float(np.percentile(timings,95)), available=counts, vocabulary=words, signer_roles=[sorted(map(int,g)) for g in roles],
                  video_ids=inventory, development_correct=development, development_total=len(dy),
                  selected=selected, test_correct=int((pred==ty).sum()), test_total=len(ty),
                  per_word={w:dict(train=int((y==w).sum()),test=int((ty==w).sum()),
                                   correct=int(((ty==w)&(pred==w)).sum())) for w in words},
                  linear_artifact_bytes=path.stat().st_size)
    Path('reports/word-vocabulary-benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(words, development,len(dy),selected,report['test_correct'],len(ty), len(y),path.stat().st_size)


if __name__ == '__main__':
    main()
