# Verified pilot results

ASLYset v1: User1–2 training, User3 validation, User4 untouched test. 24 static letters. Fixed score threshold 0.70. Selected on validation macro F1 only.

| Model | Validation macro F1 | Test accuracy | Test macro F1 | Accepted coverage | Accepted accuracy |
|---|---:|---:|---:|---:|---:|
| svm | 0.8886 | 90.98% | 0.9028 | 69.37% | 99.64% |
| random_forest | 0.7783 | 85.06% | 0.8303 | 51.75% | 98.23% |
| knn | 0.7567 | 87.30% | 0.8683 | 89.07% | 92.22% |

Selected: **svm**. Extracted 4,785/4,800 images (99.69%). Partitions after extraction: {'train': 2392, 'validation': 1195, 'test': 1198}.

Test extraction succeeded on 1,198/1,200 attempted images. The selected model correctly classified 1,090 extracted test images before confidence filtering: 90.98% on extracted images, or 90.83% counting both extraction failures as incorrect. With confidence filtering it accepted 831 test images, 828 correctly; this is 69.25% accepted coverage across all 1,200 attempted images.

These are a narrow four-signer public-dataset pilot, with only one validation and one test signer. Repeated frames are correlated within signers. This does not measure diverse webcam users, general ASL understanding, motion signs, sentences, or interpreter replacement. No threshold or estimator was changed after viewing the test set.

Live user check: user confirmed ASL B appeared once, then reported a repeat failure. This verifies one inference-path success but does not establish repeatable webcam recognition. Further live diagnosis and diverse labeled webcam recordings are needed. The browser now preserves original static uploads and uses lossless PNG webcam frames to remove extra JPEG recompression; this consistency change is not a proven cure for domain differences.

See `evaluation.json` for per-letter reports/confusion matrices, `extraction.json` for all 15 rejected extractions, and `runtime.json` for measured server request latency. Raw images and local trained weights are not pushed to GitHub; use the documented acquisition/training commands to reproduce.

Live follow-up: corrected width/height units before normalization (`wrist-max-radius-aspect-v2`). All public source images were verified square, so their feature values and model metrics are unchanged. A controlled padded-image test in `aspect-regression.json` verifies the correction; it does not establish webcam reliability. Four consented local B diagnostic photos were excluded from fitting: all models ranked B first in static mode, but SVM scores varied substantially. Mirroring did not explain the failures. Tracking drifted on repeated photos, so the app retains independent static detection for consistency. Diagnostic images and their detailed reports remain local and ignored by Git.

At the user's explicit request the browser now starts at an adjustable **30%** acceptance cutoff. This is a user preference, not an evaluated improvement. All numbers above and in `evaluation.json` still use **70%**. No 30% webcam accuracy claim is made.
