# Project handoff — resume here

Saved: **8 November 2026**. Work is paused at the user's request. The next session is for upgrades; none are running automatically.

## Current state

The local ASL lab prototype is implemented and its source/documentation/reports are published at:
https://github.com/MTJ19/Ml-Project-sign-Language-

It recognizes **24 isolated static ASL alphabet signs**, using a webcam, uploaded images, or sampled frames from browser-decodable videos. **J and Z, motion signs, words and sentence translation are deferred.** This is a lab prototype, not a validated interpreter replacement.

The user's priorities are an accurate, repeatable ML pipeline and a simple recognition view. Upgrade recognition before adding interface polish or cloud deployment. The ASL choice was deliberate because public data and references were priorities. Existing GitHub implementations were reviewed; their code and pretrained classifiers were not copied.

## Workflow already completed

1. Verified ASLYset v1 source, availability, CC BY 4.0 license and archive SHA-256. Downloaded and extracted it locally.
2. Verified four signer folders, 5,200 original images, and 50 images per signer/class. Excluded SP/FN, leaving 4,800 images across 24 letters.
3. Extracted MediaPipe's pretrained 21 hand landmarks; 4,785 images passed the exactly-one-hand rule. Logged all 15 failures.
4. Created signer-disjoint partitions: User1–2 train, User3 validation, User4 test. Counts after extraction: **2,392 / 1,195 / 1,198**.
5. Compared SVM, Random Forest and KNN. Scaling is fitted inside training pipelines. Selected **SVM using validation macro F1**, not test performance.
6. Implemented Flask/browser inference, webcam control, image/video input, labeled capture, landmark overlay, confidence feedback, no-hand/multiple-hand rejection and four-frame stability. Manual text assembly is optional and collapsed.
7. Investigated live B failures using four consented local photos. Corrected frame-aspect normalization and removed extra browser JPEG recompression. A tracking experiment drifted on repeated saved photos; the app therefore uses independent static detection for every input type.
8. Added a user-adjustable acceptance slider. Its current starting value is **30%**, explicitly requested by the user. Published baseline evaluation and the saved model's default remain **70%**.
9. Published intended source and reports through the GitHub connector. Raw data, webcam photos, local diagnostic results, model weights and the virtual environment are excluded from Git.

## Verified results and their limits

- SVM validation macro F1: **0.8886**.
- Held-out User4 accuracy: **90.98%** on 1,198 successfully extracted test images; macro F1 **0.9028**.
- At the original **70%** cutoff: **69.37% coverage**, **99.64% accuracy among accepted predictions**. These numbers do not apply to the live 30% cutoff.
- Counting the two test extraction failures as incorrect: **90.83%** over all 1,200 attempted test images.
- Server request benchmarks on the user's Apple M5 were around **22 ms per hand image** for the tested static public examples. This excludes browser encoding, network and the 180 ms polling delay; it is not measured webcam FPS.
- Eight automated checks passed after the latest code changes. A compatible WebM upload displayed stable B recognition and the overlay. An unsupported MP4 codec produced an explicit playback error.
- Live webcam B appeared, but repeat attempts were inconsistent. **Reliable webcam recognition remains unresolved.** A lower cutoff increases acceptance and can increase wrong predictions; it is not evidence of improved accuracy.

The public pilot has only four signers in a narrow setting. The test signer has already been inspected during debugging; any future tuning needs a fresh independent evaluation before making new generalization claims. Preserve the original reports as the baseline.

## Important implementation details

Pipeline: **browser frame → local Flask endpoint → MediaPipe Hands → normalized 63 features → selected SVM → cutoff → stability → displayed letter**.

- Feature version: `wrist-max-radius-aspect-v2`.
- MediaPipe normalizes x by width and y by height. Convert y into width units (`y *= height / width`) before subtracting the wrist and dividing by maximum 3D landmark radius. z is already approximately in x/width units.
- The public training images were verified square, so the aspect fix left their numerical features unchanged. Feature metadata was migrated only after checking square dimensions; all models were refitted and baseline metrics stayed unchanged.
- Orientation and handedness are retained. No browser mirroring is applied. This is not rotation invariant.
- The browser preserves original image uploads and sends lossless PNG webcam/video frames.
- Exactly one detected hand is required. No hand, multiple hands and below-cutoff predictions do not produce an accepted letter.
- Video requires four consecutive accepted predictions. Changing source/cutoff or receiving an invalid prediction clears stability.
- The browser starts at 30%, adjustable from 10–95%. Requests without a cutoff retain the model's 70% baseline. Model scores are heuristic, not calibrated probabilities or signing-correctness scores.

## Live diagnostic findings — do not confuse with training

Four B photos were manually saved under anonymous group `t1`. Their capture manifest labels them `train`, but **they were not included in classifier fitting**. Do not silently mix them into the public training/test data or treat them as an independent test set.

In static mode all three classifiers ranked B first on these photos, but SVM scores ranged roughly **21–74%**. Random Forest/KNN scored differently; four B-only photos are insufficient evidence to replace the validation-selected classifier. Detected handedness matched the public B reference, and a mirrored-feature diagnostic did not explain the failures. Tracking changed predictions on repeated photos and was disabled in the app. The aspect fix is mathematically correct and helped a controlled padded-image test, but did not resolve all saved webcam examples. Distance, landmark quality, pose differences and limited training diversity remain areas to investigate, not proven sole causes.

The detailed local-only diagnostic outputs are under ignored `data/processed/`. Do not publish the photos or their detailed reports without new authorization.

## Resume locally

From the project directory:

```sh
source .venv/bin/activate
python app.py
```

Open **http://127.0.0.1:8080/** and refresh after updates. If the previous server is still running, reuse it instead of launching a second instance on the same port. The latest local workspace already has the real trained model; a fresh GitHub clone does not include weights or data.

```sh
python -m pytest -q
# Local diagnostic analysis only; does not train on the captured photos:
python scripts/diagnose.py
```

Use pinned MediaPipe **0.10.21** because this implementation uses its bundled legacy Hands API. The tested environment is Apple silicon with Python 3.9; `requirements-lock.txt` records installed versions. Inside the agent sandbox, MediaPipe may require approved execution outside the sandbox to obtain macOS's graphics context. This is separate from ordinary local Terminal use.

To reproduce the public baseline from a fresh setup:

```sh
python scripts/download_aslyset.py
tar -xf data/raw/ASLYset.rar -C data/raw
python scripts/import_aslyset.py
python scripts/extract.py data/aslyset-manifest.csv
python -m asl_lab.train
```

These commands write local features/model/report outputs. Preserve the existing baseline reports before future experiments. Do not run extraction on the four-photo capture manifest as a replacement for the full public dataset.

## Repository and key files

- Current local tracking branch: `codex/published`, tracking `origin/main`. The original local `main` branch preserves the earlier local commit history. At publication, both histories had identical source trees.
- GitHub connector authentication works for `MTJ19` and the existing public repository. Command-line `git push` previously lacked credentials; do not assume connector authentication automatically configures CLI Git.
- `app.py`, `static/app.js`, `templates/index.html`: app and cutoff behavior.
- `asl_lab/features.py`, `asl_lab/train.py`, `asl_lab/predict.py`: preprocessing, comparison and inference.
- `scripts/diagnose.py`: local comparison on captured examples; fits candidates only on the public training partition.
- `data/aslyset-manifest.csv`: reproducible public signer partitions, generated locally.
- `data/manifest.csv`, `data/raw/collected/`: local captures; ignored by Git.
- `data/processed/landmarks.jsonl`, `models/classifier.joblib`: real local features and trained model; ignored by Git.
- `reports/evaluation.json`, `reports/RESULTS.md`: baseline numbers and limitations.
- `reports/aspect-regression.json`, `reports/tracking-experiment.json`: controlled preprocessing check and historical tracking experiment, not webcam accuracy claims.
- `docs/`: setup/workflow detail, architecture, dataset provenance, evaluation, references and demo guide.

## Next upgrade session — suggested order

1. Read this handoff and inspect the current checkout before editing. Reuse existing work; do not rebuild from scratch.
2. Reproduce the live failure on saved photos and new consented recordings. Measure detector failure, model score/label and stability separately; compare consistent resolutions, distances, palm angles, lighting and both hands. Keep preprocessing identical across training and inference.
3. Add varied, accurately labeled ASL data across classes and signers/sessions. Verify licenses and actual signer identities for any additional public dataset. Keep each signer entirely in one partition. Include unknown poses and empty/two-hand scenes as rejection probes.
4. Compare models or feature improvements on validation data, preserving the baseline. Evaluate on fresh held-out signers/sessions, report per-letter confusions, extraction losses, coverage, false accepts and latency. Do not choose an architecture based only on four B examples or lower scores solely to force a letter.
5. Once the alphabet is repeatable, extend to temporal J/Z and then selected moving signs/words using frame sequences and an appropriate temporal model. Do not describe sampled static video frames as motion-sign or sentence translation.

**Suggested next-session request:** “Read PROJECT_HANDOFF.md and continue the next upgrade, prioritizing repeatable webcam recognition and independent evaluation. Preserve the baseline, local photos and current work.”
