# J/Z motion stage — implemented, training data pending

The static 24-letter model remains unchanged. The separate **Motion lab: J and Z** section records a 2.5-second clip of landmarks from the live webcam. All hand frames must be detected, with at least eight samples and no gap over 750 ms. Images/video are not stored by motion collection; only timestamps, aspect ratio and landmarks are saved locally under ignored data/processed/motion. Stop or changing source cancels recording; no old inference result should speak after cancellation. A collection request already submitted can still finish saving its authorized sequence.

Use **Collect labeled sequence**, choose J, Z or background (NONE), enter an anonymous signer/group and partition, then perform the complete gesture once. A source that has contributed training data cannot be claimed as an independent test signer. Keep all recordings from one signer in one partition for signer-held-out evaluation. If using session IDs for a one-person pilot, state that it measures only session generalization. J/Z collection labels require correctly executed ASL motion; confirm reference technique before collecting. Collect varied speeds, distances and repetitions, plus many background/static/unrelated gestures.

The motion model is **not trained or validated yet**. Every train/validation/test partition must contain J, Z and NONE. Aim for multiple consenting signers and diverse sessions; simply reaching the code's minimum is not sufficient training diversity. No synthetic gesture rules or test fixtures are used for production recognition.

After real sequences are collected, run:

```sh
python scripts/train_motion.py
```

The trainer deduplicates across partitions by exact sequence hash, rejects groups crossing partitions, fits StandardScaler/SVM candidates only on training data, selects C using validation macro F1, and evaluates on test data afterward. Repeated tuning requires a fresh final test set. Artifacts: models/motion.joblib and ignored data/processed/motion-evaluation.json. Keep these separate from static baseline reports.

Features use 24 resampled time steps of 21 xyz landmarks, correcting image aspect and scaling against the first hand. Coordinates stay relative to the first wrist, rather than recentering each frame, so motion paths remain available. Whole-clip inference uses the separately trained model at a fixed 70% heuristic score cutoff and rejects the learned NONE class. It is not continuous sign segmentation or sentence translation, and limited background data cannot guarantee unknown-motion rejection.

Choose **Recognize motion** to classify a recorded clip. Until the trained artifact exists, the UI explicitly reports that it is untrained. Recognized J/Z can use the existing voice clips when voice is enabled. Unit/route/training tests use synthetic infrastructure fixtures only and establish no real sign accuracy. A fresh live capture and independent signer evaluation remain necessary.

## Public bootstrap model now available

SignAlphaSet v3 (Garg et al., DOI 10.17632/8fmvr9m98w.3, CC BY 4.0) was downloaded as selected archive members using HTTPS byte ranges. Individual uncompressed sizes and CRCs were checked; no full-archive SHA verification is claimed. The archive directory contained 300 AVI videos, including ten J and ten Z clips, and no signer/session metadata. Sampled target clips were inspected as temporal contact sheets; source labels remain publisher annotations, not independent expert ground truth.

Downloaded 80 clips: J/Z plus A/I/D/K/S/U background candidates. Extracted 76 valid contiguous landmark sequences: J=10, Z=10, NONE=56. Four background clips failed minimum sequence requirements. All 76 accepted video hashes are unique. Changes from source are subset selection, 10fps landmark extraction, rejection of inadequate clips, temporal resampling and conversion of non-J/Z source classes to background. Preserve source attribution and license when distributing derivatives.

Trained a balanced RBF SVM with C=1 fixed before training; it is a **bootstrap prototype**, not independently evaluated. All 76 source clips are training data. No signer identities, independent partitions, or accuracy figures were invented. The original strict scripts/train_motion.py remains available for later genuinely partitioned collection and evaluation.

Artifacts: models/motion-public-bootstrap.joblib and active models/motion.joblib. Extraction report/metadata: ignored data/processed/motion-public/. Public aggregate report: reports/upgrades/motion-bootstrap.json. Download reproduction: python scripts/download_motion_subset.py; extraction/bootstrap training: python scripts/bootstrap_public_motion.py (writes a separate candidate, not active weights). A full archive download is unnecessary for this subset.

The live endpoint was smoke-tested on one source training clip per class (J/Z accepted, background rejected). These are integration checks, not held-out accuracy. Existing 13 Python tests and voice/evaluation logic checks passed. The existing static classifier, main app script, voice code and A audio checksums are unchanged. Only motion result wording and its section notice were updated to mark it experimental. Refresh, choose **Recognize motion**, then record fresh J/Z/background clips for the next independent webcam check; do not fit those same clips before evaluating them.

## J lateral-inversion diagnosis and two-clip adaptation

User hypothesized lateral inversion. Evaluated the same four J and one Z validation clips with x mirrored and unmodified: mirroring left every J's top class as Z, while slightly lowering Z score. No camera/display orientation change was made.

With explicit user approval, fitted a separate candidate using the first two saved J clips in capture order alongside the 76 public sequences. Fixed C=1 and adaptation weight=3 before evaluation. Remaining two J clips and the single Z clip were excluded from fitting. Candidate ranks both reserved J as J (scores ~0.780 and ~0.656), instead of baseline Z (~0.822 and ~0.819). However the reserved Z score drops from ~0.767 to ~0.528. At the existing 70% motion cutoff, only one reserved J is newly accepted and the previously accepted Z would abstain. Candidate **not activated** to preserve working Z behavior; active motion.joblib is unchanged. Candidate weights are models/motion-J-candidate.joblib; sample roles/detailed comparison stay local in data/processed/motion-J-adaptation.json. Reproduction: scripts/adapt_motion_j.py. These same-user comparisons are development diagnostics, not independent accuracy.

Next: collect additional varied correctly labeled Z and background clips for balanced local adaptation, and reserve fresh later J/Z/background recordings for evaluation. Do not lower the motion cutoff just to make these three examples pass or claim mirroring fixes them. Static model, voice and camera behavior remain unchanged.

## Recorder usability fix

The initial recorder cancelled on the first no-hand/multiple-hand response. It now skips missing detections, continues the recording, and displays usable/missed frame counts. At completion it still requires at least eight valid frames, at least 0.7 seconds of detected coverage, and no gap over 750 ms; incomplete clips are rejected with a specific message rather than saved with fabricated landmarks. Stop/source/request failures still cancel. Motion feature thresholds, classifier weights and static recognition are unchanged. Node UI tests verify a brief missed detection still saves, a long gap does not post/save, and Stop cancels. Refresh to load motion.js version 3.

## Balanced local adaptation activated

User supplied more Z and background clips. Selected adaptation training now contains the two previously approved J clips, three new Z and three background clips alongside the public source training data. The original first Z remains reserved. Remaining two J, three Z and three background clips were excluded from fitting. Fixed C=1 and sample weight=3 before fitting; exact duplicate sequences cannot cross roles. Development comparison: baseline accepted 0/2 J, 1/3 Z and correctly rejected 1/3 backgrounds; candidate accepted 2/2 J, 3/3 Z and rejected 3/3 backgrounds at the unchanged 70% motion cutoff. These eight same-user, already-inspected clips are not independent evaluation or evidence of 100% general accuracy.

Candidate is now active in models/motion.joblib; rollback weights are models/motion-pre-balanced.joblib. Static classifier checksum is unchanged; no camera mirroring or voice changes. Local detailed roles/results: data/processed/motion-balanced-adaptation.json; public aggregate: reports/upgrades/motion-balanced-summary.json; reproduction: scripts/adapt_motion_balanced.py. All 13 Python checks and recorder/voice checks pass. Next: test new J/Z/background recordings from a fresh session before fitting them; do not reuse these 16 clips as a new independent test.

## Ready-hand timing and uncertain feedback

User reported live J abstentions, position-sensitive Z and too-few-frame errors. Recorder now waits up to five seconds for the first detected hand before starting its 2.5-second capture window. During motion activity only, the recognition loop's scheduling delay drops from 180 ms to 40 ms to increase sample availability; actual FPS still depends on detection/encoding speed. Static idle scheduling is unchanged. Minimum eight valid frames, 0.7-second span and maximum 750 ms gap remain unchanged. Motion abstentions now show the top candidate, its score, required 70% cutoff and frame count, distinguishing J below threshold from background classification. Weights/cutoffs were not altered. Node tests cover initial no-hand preparation, capture timer start and timeout reset; 13 Python checks pass. Fresh live clips are still needed to verify whether J itself has improved; recorder changes alone are not a model-accuracy fix.

## Timing-robust temporal matching

User reported persistent live J abstention and intermittent Z despite the small reserved comparison. Stopped requesting more training clips and investigated offline. Synthetic start/end holds on existing reserved sequences exposed timing sensitivity: active SVM correct decisions across eight source development clips were original 8/8, start-pause 5/8, end-pause 7/8, both 5/8. Pause-augmented SVM training regressed original examples (6/8), so it was not activated.

Added dynamic-time-aligned nearest-template matching trained on the exact same previously approved public+local training roles. It aligns movement timing rather than requiring fixed frame-to-frame phase. Temperature is fixed at 0.03; normalized similarity scores are heuristic, not calibrated confidence. Reserved originals remain 8/8; start holds 8/8, end holds 7/8, both holds 7/8. These are already-inspected same-user development clips and synthetic perturbations, NOT independent or live recognition accuracy. Unknown-motion rejection is not guaranteed. No new user clips were fitted.

The aligned candidate is now active. Rollback is models/motion-pre-aligned.joblib. Static classifier, voice code, A audio and main app script were hash-checked unchanged at activation. Reproduce candidate with scripts/check_temporal_match.py; scripts/robust_motion.py records the unpromoted augmentation experiment. Details remain ignored data/processed/; aggregate report is reports/upgrades/motion-time-alignment.json. Fourteen Python tests pass, including repeated-pause alignment. The live endpoint was smoke-tested on existing development examples, not fresh user attempts. New live reliability remains unproven; do not describe J as definitively fixed.

## Fresh live J still unreliable — review without re-recording

User reported Z mostly recognized and J rarely recognized after time-aligned update. This supersedes any implication that offline timing gains fixed live J. No further model replacement was made on this report. Added **Save last attempt for review**: after Recognize motion, choose the correct label and click it to persist that exact validated landmark sequence locally under validation group motion_review, without repeating the gesture or adding training data. Only explicit button activation saves; refreshing discards the browser-held last attempt. Successful save disables the button to avoid duplicates. This diagnostic group is not proof of a different signer or session. The next investigation should use a genuinely failed fresh J attempt rather than recycling the eight old development clips.

## Optional automatic J/Z in the main webcam view

Added an **Auto J/Z (experimental)** checkbox beside the main camera controls. Default off preserves static-only behavior. When enabled on a webcam, it keeps a rolling 2.6-second landmark buffer, requires at least eight frames and two seconds of coverage, checks for meaningful 2D movement relative to initial hand size, and queries the existing motion model at most every 600 ms with only one request in flight. Two agreeing J/Z results at the unchanged 70% motion cutoff are required before the main display shows the motion letter for up to 1.5 seconds. Similarity scores are heuristic. This gate is generic motion detection, not a hard-coded J/Z gesture classifier.

No-hand/source changes/disable clear pending recognition; stale asynchronous responses are ignored. Manual Motion lab recording and static evaluation pause automatic inference. Evaluation remains a static-letter protocol, so automatic J/Z is intentionally inactive during those trials. Uploaded images/videos do not use Auto J/Z in this version. With voice enabled the displayed motion letter uses the existing female clip path. Static model/voice assets and motion-model weights are unchanged. The main loop requests frames faster only when auto mode is enabled.

Node tests exercise no inference on stationary hands, repeat agreement, disable/no-hand clearing and stale-response cancellation. Existing voice/evaluation/recorder suites and 14 Python tests pass. Live rolling segmentation remains experimental and may miss gestures or accept false positives; these logic checks do not establish recognition accuracy. Static letters can appear during a gesture before motion confirmation. Disable Auto J/Z to restore static-only behavior. Motion lab remains available for deliberate recording and exact-attempt review.

## Distinguishing static P from unconfirmed Z

User reports J recognized in Auto mode but Z appears as P. The motion classifier's classes are only J/Z/NONE: P originates in the parallel static classifier while motion is unconfirmed, not a literal Z→P temporal output. Auto mode now detects recent movement and temporarily suppresses static display/speech, showing “Reading motion — static letters paused” with the latest motion candidate/score. A confirmed J/Z result still requires the unchanged model cutoff and agreement; no forced Z mapping is added. Static recognition resumes after movement settles, and static-only mode is unaffected. This is a segmentation/presentation fix, not proven improvement of Z classification. Tests cover movement gating, idle recovery, no-hand/source cancellation and voice behavior; 14 Python tests pass. Verify live rolling behavior before declaring Z reliable.

## Confirmed motion ending-pose latch

User reports delayed J/Z and Z reverting to static P. The old confirmed-motion display expired after 1.5 seconds regardless of the held pose. A confirmed J/Z now persists while the ending handshape remains similar, with a minimum 1.5-second hold. It clears on a changed normalized handshape after that hold, no hand, disable or source reset. Translation of the hand alone does not count as a new shape. This prevents the static ending-pose label from immediately replacing confirmed Z; it does not force a motion prediction or improve classifier accuracy. New motion confirmations can still update the letter. Existing motion/score/agreement gates remain unchanged. Tests cover pose persistence, translation tolerance and handshape-change release. Full-gesture collection plus repeated confirmation still causes several seconds of latency.


### Completed-gesture Auto J/Z update

Supersedes rolling-window repeated agreement above. Auto J/Z now segments movement and waits for a 450 ms ending hold before classifying one completed sequence. Uncertain/background/error outcomes stay visible while the ending pose is held, preventing static P from replacing unresolved motion. A changed handshape or removing the hand resets the outcome. Incomplete recordings are rejected; motion threshold remains 70%, and model weights/static recognition are unchanged. E uses regenerated Samantha audio (e.) with cache refresh. Fourteen Python tests and completed-gesture/voice JavaScript checks pass. Actual pronunciation and live J/Z reliability still require user verification. Refresh the app, enable Auto J/Z, complete the gesture, and briefly hold still. Changes remain local.
