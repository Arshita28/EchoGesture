# EchoGesture

[Open the college ML showcase in Google Colab](https://colab.research.google.com/github/MTJ19/EchoGesture/blob/main/notebooks/EchoGesture_Showcase.ipynb)

The notebook supports uploaded letter images, J/Z videos, 81-word videos, model inspection, and labelled evaluation. Select the 2026.07 Colab runtime (Python 3.12). Sign into GitHub to access this private repository.

EchoGesture builds on the existing ASL Landmark Lab. **Phase 1** is the 24-letter static recognizer. Further phases add word-sign data, temporal recognition, live captions and measured local inference. The final website or app interface is undecided. See [the development roadmap](docs/EchoGesture_Development_Roadmap.docx) and [continuation notes](docs/ECHOGESTURE_START.md).

**Continuing this project? Read [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) first** for the saved workflow, current state, diagnostic findings and next upgrade steps.

Local ML lab for **24 static ASL fingerspelling letters** (A–Y excluding J; Z also excluded). Webcam, uploaded images and browser-playable videos share the same normalized-landmark classifier. The alphabet workspace recognizes static signs. The separate word workspace processes short temporal video clips.

## Latest local studio

The completed 80-word candidate is included in `models/word80-priority-wlasl-citizen-svm.joblib`.
After installing the requirements, launch it with:

```sh
ECHOGESTURE_WORD_MODEL=word80-svm python app.py
```

Open http://127.0.0.1:8080/words for its current vocabulary or /sign for the alphabet studio.
The default configuration loads the completed 80-word model.
Recorded test scores vary by dataset and vocabulary. See the reports directory for measured comparisons.
Raw videos, derived landmark caches and third-party archives require separate acquisition under their dataset licences.
The landing-page template licence and required credit links are retained.

## Word recognition

The default Word Studio loads the trained **80-word model**.
Run `python app.py`, then open http://127.0.0.1:8080/words.
The vocabulary shown on the page comes directly from the loaded checkpoint.
Use **Speak result** to hear a prediction, or enable **Auto voice**.

The 80-word checkpoint is the default experimental recognizer.
Evaluation reports record dataset-specific accuracy and limitations.
The earlier five-word model remains available with `ECHOGESTURE_WORD_MODEL=prototype`.

## Run

Python 3.9–3.12 recommended. The current environment was installed with Python 3.9 on Apple silicon. Use the pinned MediaPipe version: newer releases may remove `mp.solutions`.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8080 for the landing page; choose **Start signing** to open `/sign` and allow browser camera access. The app displays “No trained classifier” until a real model is trained. Do not load joblib files from untrusted sources. The Flask development server binds only to this computer.

The signing workspace uses a fixed **30%** acceptance setting, as requested during live testing. This is a provisional live setting, not a validated optimum; lower cutoffs can accept more wrong signs. The saved model and published baseline evaluation remain at **70%**. No-hand/multiple-hand rejection and four-frame stability still apply. Webcam reliability is not established by the public-dataset pilot.

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

## Current application and project documentation

The main workspace recognizes static letters and supports manual message composition with spaces, deletion, copying and full-message speech. J/Z remains unreliable and is available only in experimental Development tools (`/sign?tools=1`). The original benchmark does not measure the later personalized model.

- [Project report](docs/PROJECT_REPORT.md)
- [Demo checklist and viva preparation](docs/DEMO_CHECKLIST.md)
- [Live evaluation protocol](docs/EVALUATION_MODE.md)

## Organized ML datasets

Start with [datasets/README.md](datasets/README.md) and the [dataset report](datasets/DATASET_REPORT.md) for audited source counts, preprocessing, exclusions and actual training roles. Raw media lives in `datasets/raw/`; derived data in `datasets/preprocessed/`. Historical `data/` paths continue to work through a compatibility link. The local SHA-256 inventory can be verified with `python scripts/verify_dataset_inventory.py`.

The public landing page includes original procedural particle artwork. Optional reference-derived local geometry is ignored by Git; see [frontend provenance](docs/FRONTEND_REFERENCE.md). Private datasets and trained weights are not shipped.
