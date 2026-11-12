# Sign Studio: landmark-based ASL fingerspelling recognition

## Abstract
Sign Studio is a local prototype that recognizes 24 static ASL fingerspelling letters from webcam frames, uploaded images and browser-decodable videos. MediaPipe supplies pretrained hand detection and 21 hand landmarks. Classical machine-learning classifiers predict letters from numeric hand features. The browser stabilizes predictions, presents uncertainty, supports manual message composition and plays spoken output. Experimental J/Z motion recognition remains unreliable and is excluded from the main signing flow.

## Objective and scope
The project investigates whether compact landmark features can support a practical classical-ML interface within a lab-course scope. It supports isolated letters and manually assembled messages, not continuous sign-language translation, grammar understanding, ISL recognition, or replacement of interpreters. The user controls repeated letters and word boundaries. Words typed into the composer are not model predictions.

## Architecture and methods
The landing page (`/`) links to the signing workspace (`/sign`). The browser sends sampled frames to local Flask endpoints. MediaPipe detects a single hand, returning 21 x/y/z coordinates. Height-normalized y coordinates are converted to image-width units before wrist subtraction and maximum-radius normalization. Orientation remains represented; the features are not fully rotation invariant.

The original benchmark compares SVM, Random Forest and KNN on 63 normalized landmark coordinates. The later geometry model augments those with 210 pairwise landmark distances and 15 joint-angle cosine features, for 288 features. The deployed static classifier is a geometry SVM personalized with selected consented local examples. Its current webcam performance has not been independently quantified.

The signing UI accepts predictions at a fixed 30% score setting and requires four consecutive agreeing accepted frames before displaying a video letter. This setting was chosen for usability during local testing, not established as an optimal threshold. Scores are not guarantees of correctness. Invalid detections clear the result; unfamiliar gestures can still receive confident wrong predictions.

## Dataset and experimental protocol
The original ASLYset pilot uses User1–2 for training, User3 for validation and User4 for testing. Training compares models and selects on validation macro F1. Signer and duplicate-image checks prevent partition overlap. Of 4,800 attempted images, 4,785 were extracted: 2,392 training, 1,195 validation and 1,198 test examples. There is only one validation and one test signer, limiting generalization claims.

Subsequent webcam adaptation and motion experiments are development work. Already-inspected local examples and synthetic timing variations must not be presented as an independent test set. Fresh-session trials assess repeatability for the current user; new consenting signers are needed to test generalization.

## Original benchmark results
These values come from `reports/RESULTS.md` and `reports/evaluation.json`. They describe the original public-dataset models, not the currently personalized model or general live-webcam accuracy. Acceptance metrics use the original 70% threshold.

| Model | Validation macro F1 | Test accuracy before filtering | Test macro F1 | Accepted coverage | Accuracy among accepted predictions |
|---|---:|---:|---:|---:|---:|
| SVM | 0.8886 | 90.98% | 0.9028 | 69.37% | 99.64% |
| Random Forest | 0.7783 | 85.06% | 0.8303 | 51.75% | 98.23% |
| KNN | 0.7567 | 87.30% | 0.8683 | 89.07% | 92.22% |

SVM was selected. It correctly classified 1,090 of 1,198 extracted test images. Counting two extraction failures among all 1,200 attempted test images gives 90.83%. At the 70% threshold it accepted 831 extracted examples, of which 828 were correct. The high accepted accuracy therefore comes with substantial abstention; it is not accuracy over all attempts.

## Implemented application features
- Webcam, image and supported video input; detected-hand overlay.
- Stable static-letter display and explicit uncertain/no-hand/multiple-hand states.
- Manual append, spaces, deletion, clear and copy controls.
- Fixed Samantha letter audio; full-message speech through an available local named female English voice. Message speech suppresses overlapping letter announcements.
- Separate development tools for labeled capture, experimental motion sequences and trial-level evaluation.
- Readable downloadable evaluation reports with wrong/uncertain counts, coverage and rejection checks. A three-second trial is one observation; correlated video frames are not counted as independent tests.

## Motion recognition: incomplete work
J and Z require temporal trajectories. A separate experimental pipeline resamples landmark sequences and compares time-aligned templates. Small development comparisons improved, but the user continues to report unreliable live J/Z recognition. Automatic motion arbitration also suppressed static results during ordinary hand movement; it is now disabled in the main webcam. Motion tools remain available for research. J/Z is not a completed feature or a claimed success metric.

## Verification and limitations
Automated checks cover preprocessing, leakage protection, motion sequence handling, evaluation logic, voice state and browser controls. Fake-camera browser tests validate interface behavior only; they do not measure recognition accuracy. The user completed a message-flow check, but no new quantitative live evaluation report has been supplied. Leave current webcam accuracy unreported until measurements are recorded.

Lighting, distance, hand orientation, occlusion and signer differences remain important failure sources. Public-dataset results do not establish real-world accessibility effectiveness. Speech availability depends on the installed browser/system voices. Full messages remain local when the supported local voice is used. Raw captures and trained weights stay outside Git.

## Reproduction and demonstration
Follow the repository README for installation and baseline acquisition/training. Start `python app.py`, open `/sign`, start the camera and compose a short message. Use `/sign?tools=1` to record fresh labeled evaluation trials and download the readable report. See `docs/DEMO_CHECKLIST.md` for a repeatable demonstration and `docs/EVALUATION_MODE.md` for the measurement protocol.

The local landing artwork reproduces reference geometry for design review. Asset provenance is documented in `docs/FRONTEND_REFERENCE.md`; publication is not part of this milestone.

## Next work
1. Quantify current static-model performance with fresh trials across all 24 letters and rejection cases.
2. Diagnose confusions using a designated adaptation set; reserve separate examples for comparison.
3. Improve and independently test J/Z before re-enabling automatic motion integration.
4. Finalize presentation and report with actual live metrics; polish the frontend after reliability checks.

## Evidence sources
- `reports/RESULTS.md`, `reports/evaluation.json`: original benchmark and per-letter results.
- `reports/extraction.json`: extraction accounting.
- `reports/upgrades/`: development comparisons, not independent live accuracy.
- `docs/dataset.md`, `docs/references.md`: dataset provenance and technical references.
- `PROJECT_HANDOFF.md`: implementation history and current limitations.
