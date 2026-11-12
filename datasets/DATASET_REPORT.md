# Dataset provenance, preparation and usage report

Audit date: **9 November 2026**. Scope: actual local files used or retained by Sign Studio. Counts are snapshots; future collection changes them. No synthetic examples were generated during this audit.

## 1. Summary

| Source | Raw material present | Prepared data | Actual usage |
|---|---|---|---|
| ASLYset v1 | 5,200 PNG images; original archive and annotations retained | 4,800 selected alphabet images → 4,785 valid feature rows | Original static SVM/RF/KNN benchmark; public training portion reused in later static models |
| Local webcam static captures | 71 JPG images | Extracted on demand by diagnostic/adaptation scripts | Latest adaptation record lists 34 training images and 23 diagnostic images; 14 not listed in that record |
| SignAlphaSet v3 subset | 80 AVI clips | 76 usable sequences: 10 J, 10 Z, 56 background | Public motion bootstrap training; not a held-out accuracy set |
| Local webcam motion | No original webcam video retained by the motion recorder | 26 saved landmark sequences: 4 J, 16 Z, 6 background | 8 selected adaptation training, 8 reserved development comparisons, 10 additional saved sequences outside those role lists |
| Local upload debugging | 1 MP4 | No training rows attributed | Compatibility/debug input; do not count as training data |
| Proposed phrase datasets | No phrase training videos acquired | No phrase features/model | How2Sign and ASL Citizen researched only; not used for training |

“Preprocessed” includes detector outputs and historical diagnostic reports. Saved motion frame coordinates are transformed/resampled when features are built; they are not already independent labeled frame samples. Models are stored separately under `models/`.

## 2. ASLYset v1: raw source and subset

- Creator: Miguel Rivera; published 31 October 2019.
- Publisher: https://data.mendeley.com/datasets/xs6mvhx6rh/1
- DOI: https://doi.org/10.17632/xs6mvhx6rh.1
- License recorded in the acquired publisher metadata: CC BY 4.0.
- Metadata snapshot: `documentation/aslyset-metadata.json`.
- Original archive: `raw/ASLYset.rar`, 872,115,989 bytes. Recorded SHA-256: `e50202c30dfff4795b88762b765bbeb357e184c13fe22b8f6e0d836d02235c80`.
- Raw images: 5,200 square 416×416 RGB PNG files, four signer folders. The extracted source also contains 5,201 TXT and 99 INI files; these are **not additional images**.

The source has 24 static alphabet classes plus SP/FN, 50 images per class per signer. We selected the 24 static letters (A–I, K–Y), excluding SP/FN: 400 images remain retained in raw data but excluded from the 4,800-row alphabet manifest. J/Z are not supplied as static recognition classes. Original annotations were retained, but YOLO boxes are not inputs to our landmark classifier.

### Split and extraction accounting

| Split | Signers | Selected raw images | Valid extracted rows | Extraction failures |
|---|---|---:|---:|---:|
| Train | User1, User2 | 2,400 | 2,392 | 8 |
| Validation | User3 | 1,200 | 1,195 | 5 |
| Test | User4 | 1,200 | 1,198 | 2 |
| Total | Four | 4,800 | 4,785 | 15 |

Original signer identities define the partitions. No random adjacent-frame split is used. Extractor failures and per-class counts are recorded in `preprocessed/landmarks.extraction.json` and `documentation/baseline-extraction.json`. The original benchmark uses validation macro F1 for model selection; its test report is preserved in `documentation/baseline-evaluation.json`.

### Static preprocessing and later changes

1. Read the source RGB image and run pretrained MediaPipe Hands in independent static-image mode.
2. Require exactly one detected hand; rejected images do not become training vectors.
3. Extract 21 x/y/z landmarks. Convert y to width-based units using height/width, subtract the wrist and divide by the maximum 3D landmark radius. This yields 63 numbers.
4. Store normalized rows with label/group/split metadata in `preprocessed/landmarks.jsonl`.
5. Later geometry models compute 210 pairwise landmark distances and 15 joint-angle cosine features in addition to the 63 coordinates (288 features total). This transform is applied in the model pipeline, not saved as a second raw image dataset.

The aspect correction fixes rectangular webcam geometry; source ASLYset images are square, so their original values are unchanged by that correction. No claim of full rotation invariance is made. No raw-image augmentation or image alteration was introduced by this folder reorganization. Original images and archive remain retained.

Attribution for derivatives: Derived normalized landmarks from ASLYset v1, Miguel Rivera (2019), DOI 10.17632/xs6mvhx6rh.1, CC BY 4.0. Changes: selected static classes, extracted/normalized landmarks, assigned signer-separated partitions and added geometric model features.

## 3. Local static images: collection versus actual training roles

Location: `raw/collected/`; original capture manifest: `manifest.csv`.

| Letter | Images | Letter | Images |
|---|---:|---|---:|
| A | 5 | B | 4 |
| C | 10 | D | 5 |
| E | 5 | G | 4 |
| I | 4 | M | 8 |
| P | 8 | Q | 3 |
| S | 6 | U | 6 |
| X | 3 | Total | 71 |

These are user-provided webcam frames for collection, diagnosis and explicitly selected personalization. They have no public redistribution license. Keep photos, identity information and detailed path inventories local.

Original capture manifest labels are 14 `train` and 57 `validation`. **These are capture-time labels, not the final model-fitting ledger.** With user authorization, selected examples were later reassigned for adaptation. `preprocessed/local-adaptation.json` lists the latest actual roles: **34 training**, **23 reserved diagnostic** images. The other **14** captured images are not listed in those latest role sets; do not assume all were successfully extracted, used, or independently evaluated.

Adaptation scripts deduplicate image hashes and select examples before fitting. They combine selected local frames with original public training rows. Historical `adaptation-pre-*.json` and diagnostic reports document prior experiments. Reserved local diagnostics are correlated same-user/session examples and have been inspected; they are **not independent generalization evidence**. Do not equate 71 images to 71 fresh test cases.

## 4. Public motion subset: SignAlphaSet v3

- Publisher: https://data.mendeley.com/datasets/8fmvr9m98w/3
- DOI: https://doi.org/10.17632/8fmvr9m98w.3
- Creators recorded in project provenance: Bindu Garg, Manisha Kasar, Achyut Kashyap, Amber Vats, Gunjan Sharma and Aditya Hange.
- License recorded in metadata: CC BY 4.0.
- Metadata: `documentation/signalphaset-metadata.json`.

Only **80 AVI members** were downloaded from ASL_dynamic.zip, using selective ZIP byte-range extraction and member size/CRC validation. The complete approximately 1.25 GB archive was not downloaded or SHA-verified locally. A member index is retained under `raw/signalpha/`. Do not claim the entire source dataset was used.

J/Z clips retain their target labels. Selected A/I/D/K/S/U static/background clips are mapped to `NONE` for motion rejection; original source labels remain in the sequence records. This relabeling is task-specific: NONE does not mean the video contains no hand or no valid static sign.

Extraction samples at approximately 10 fps, runs independent hand detection and retains a sufficiently complete contiguous segment. Four background clips failed minimum usable-frame requirements, leaving **76 sequences: 10 J, 10 Z, 56 NONE** in `preprocessed/motion-public/sequences.json`. The extraction report lists the failures. Feature construction corrects aspect ratio, uses initial-wrist/hand-scale normalization to preserve trajectory, and resamples to 24 steps of 63 values (1,512 temporal features).

All 76 are bootstrap training examples. No verified signer identities were available for these selected clips, so they cannot support signer-independent test claims. Active motion inference later changed from an SVM candidate to a time-aligned template matcher using the same designated training roles; synthetic pause tests are development stresses, not new independent recordings.

## 5. Local motion sequences

Location: `preprocessed/motion/`. The recorder saves timed 21-point landmark frames and label/group/split/version metadata, **not original camera video**. Raw webcam videos cannot be reconstructed from these files. Do not fabricate a raw-video counterpart.

Current inventory: **26 sequences: J=4, Z=16, NONE=6**. All files retain `validation` as their original capture-time split. The explicit adaptation ledger `preprocessed/motion-balanced-adaptation.json` overrides that label for eight authorized training sequences. Eight separate sequences are recorded as reserved development comparisons; ten later saved sequences are outside those two lists. The saved label alone must not determine which samples were actually fitted.

Small development results do not resolve current live J/Z failures. Motion recognition remains experimental and is disabled in the main static webcam. No training was performed during organization.

## 6. Inventory, integrity and interpretation

`inventory/files.csv` records relative path, byte size and SHA-256 for every real file under raw/preprocessed at migration. It intentionally excludes compatibility symlinks to prevent double-counting. `inventory/summary.json` contains aggregate counts.

`data -> datasets` and `datasets/processed -> preprocessed` preserve historical scripts and stored paths. Original manifests remain beside raw/preprocessed because paths such as `raw/collected/...` are relative to their parent folder. Do not move these CSVs without rewriting their paths.

Original public evaluation, later personalized validation and local diagnostics must be reported separately. A file's presence does not prove training use; use the actual role ledgers. The current UI's 30% threshold also differs from the original benchmark's 70% threshold.

## 7. Reproduce the original public pipeline

From the project root, the historical commands still resolve through compatibility aliases:

```sh
python scripts/download_aslyset.py
# Extract only if rebuilding an empty raw folder:
tar -xf data/raw/ASLYset.rar -C data/raw
python scripts/import_aslyset.py
python scripts/extract.py data/aslyset-manifest.csv
python -m asl_lab.train
```

Training can replace existing model artifacts: preserve the current model before rerunning. The inventory verifier is read-only and can be run safely without retraining. Consult the motion scripts and `docs/MOTION_STAGE.md` for motion history rather than running candidate promotion scripts blindly.
