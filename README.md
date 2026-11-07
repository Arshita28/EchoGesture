# ASL Landmark Lab

Local ML lab for **24 static ASL fingerspelling letters** (A–Y excluding J; Z also excluded). Webcam, uploaded images and browser-playable videos share the same normalized-landmark classifier. Video processing recognizes isolated static signs in sampled frames, not motion signs, words, or sentences.

## Run

Python 3.9–3.12 recommended. The current environment was installed with Python 3.9 on Apple silicon. Use the pinned MediaPipe version: newer releases may remove `mp.solutions`.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8080 and allow browser camera access. The app displays “No trained classifier” until a real model is trained. Do not load joblib files from untrusted sources. The Flask development server binds only to this computer.

The browser acceptance slider starts at **30%**, as requested during live testing. This is a provisional live setting, not a validated optimum; lower cutoffs can accept more wrong signs. The saved model and published baseline evaluation remain at **70%**. No-hand/multiple-hand rejection and four-frame stability still apply. Webcam reliability is not established by the public-dataset pilot.

The local workspace now includes a genuinely trained 24-class SVM. The public repository contains source and reports, with data/weights reproducible using the steps below. Verified pilot test accuracy: **90.98%** on extracted images from the held-out signer; see [full results and limitations](reports/RESULTS.md). This is not a general webcam accuracy claim. `requirements-lock.txt` records the tested Apple-silicon Python 3.9 environment.

## Dataset and training

Collect consented images using the interface, or import ASLYset (see [dataset provenance](docs/dataset.md)). Use a stable signer ID, not a new ID per frame/session. Keep a person's recordings in one partition. The minimum pilot needs at least one different signer in each of train, validation and test, with every selected letter represented in all three. Prefer several signers per partition for a stronger study.

```sh
python scripts/download_aslyset.py
tar -xf data/raw/ASLYset.rar -C data/raw
python scripts/import_aslyset.py
python scripts/extract.py data/aslyset-manifest.csv
python -m asl_lab.train
```

For your own collected images instead:

```sh
python scripts/extract.py data/manifest.csv
python -m asl_lab.train
```

The CSV columns are `path,label,group,split`; paths are relative to the CSV. Partitions are `train`, `validation`, `test`. Extractor failures are saved beside the landmark JSONL. Training rejects groups or identical image hashes crossing partitions and missing classes. It compares SVM, Random Forest and KNN, selects only on validation macro F1, and writes `models/classifier.joblib` and `reports/evaluation.json`. Restart/refresh the app after training to update the model badge.

## Project layout

- `asl_lab/`: normalization, MediaPipe extraction, classification and training
- `app.py`, `templates/`, `static/`: local Flask/browser app
- `scripts/`: public data acquisition/import and feature extraction
- `data/raw/`, `data/processed/`, `models/`: local images, features and trusted artifacts (excluded from Git)
- `reports/`: real evaluation reports; test fixture scores are never published as sign accuracy
- `docs/`: [architecture](docs/architecture.md), [dataset](docs/dataset.md), [evaluation](docs/evaluation.md), [demo](docs/demo.md), [references](docs/references.md)

```sh
python -m pytest -q
```

No full-language translation or interpreter replacement is claimed. Predictions can be wrong, including confident mistakes on unfamiliar gestures.
