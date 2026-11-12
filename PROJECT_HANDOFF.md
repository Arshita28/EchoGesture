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

## Latest continuation — geometry upgrade

The project is active again following reported V/C and B/I/F/M/S/U confusions. Read `docs/UPGRADE_LOG.md` and `reports/upgrades/geometry-validation.json` before continuing. A validation-selected geometry SVM is now active locally; original model weights and original test reports are preserved. Validation macro F1 rose to 0.9430, with a U recall regression explicitly recorded. More labeled webcam examples are being collected; none were used in fitting. No new independent test accuracy is claimed.

### Local personalization follow-up

A/M/S adaptation was subsequently authorized and is now active locally. Nine selected webcam frames were reassigned to training; detailed roles/results stay under ignored data/processed/local-adaptation.json. Public validation macro F1 is 0.9452, but two errors remain among eight same-session reserved frames. No independent webcam accuracy claim is warranted. Prior geometry weights are saved at models/geometry-pre-adaptation.joblib. Collect a fresh A/M/S recording and check C before calling the fix reliable. See docs/UPGRADE_LOG.md.

### Complete alphabet test and D/P follow-up

Read docs/ALPHABET_TEST_LOG.md for the user's full working/failing list. Local adaptation now also includes selected D/P training frames. Active model: geometry_local_adaptation_dp; previous weights preserved at models/pre-dp-adaptation.joblib. Reserved D frames improved; P still has below-cutoff cases. Public validation macro F1 is 0.9460, not fresh webcam accuracy. The user is collecting remaining E/Q/U/X/G and uncertain C examples. Do not restart earlier experiments or claim these failures are resolved. Local details are recorded in docs/UPGRADE_LOG.md and ignored data/processed/local-adaptation.json.

### Latest active model — full local adaptation

User supplied the remaining capture batch. Active local model is geometry_local_adaptation_full; rollback is models/pre-full-adaptation.joblib. Read the latest docs/UPGRADE_LOG.md entry: public validation macro F1 0.9491; reserved E/Q/U/X/C/G examples improved, but P abstentions and M/S errors remain. Captures were partly reassigned to training with reserved correlated diagnostics; exact roles stay local in data/processed/local-adaptation.json. Next: fresh-session alphabet regression, not more claims from these same frames. Local photos remain private; original baseline reports remain intact.

### Voice output added

Optional browser speech output is implemented in static/voice.js and integrated into recognition. Refresh and click Enable voice. Accepted stable letters speak once; no-hand for one second rearms the same letter. Stop, source changes, cutoff changes and disabling voice cancel/reset speech. Uncertain/no-hand/multiple-hand states never speak a guessed letter. Node behavior checks and JavaScript syntax checks passed alongside the nine Python checks; actual audible output depends on the live browser/system voice and requires a manual check. No model, cutoff or training-data changes were made for voice output.

### Female voice correction

Browser voice preference still fell back to male speech. Generated 27 local WAV clips (A–Z plus “Voice enabled”) with installed macOS Samantha and changed production playback to these fixed clips. No browser default substitution in the normal Audio path. Cache-versioned script URLs require page refresh. Audio files validated, local audio endpoint returned 200, mocked clip playback/dedup/mute checks passed. User still needs to confirm audible playback. This is Samantha, not Siri's exact voice.

### Latest local I adaptation

Active model is now geometry_local_adaptation_I. Three of four new I captures were trained; one reserved frame scored I at 0.8393. Public validation macro F1 remains 0.9491 and public I/Y recalls stayed unchanged. No Y webcam captures were saved, so fresh local Y regression is still needed. Rollback: models/pre-I-adaptation.joblib. See latest upgrade log. This local change has not been pushed since the earlier GitHub publication.

### Latest voice repeat behavior

At the user's request, speech re-arms immediately whenever the displayed recognition clears. I → blank → I speaks again; uninterrupted I speaks once. No other letter or one-second hand-removal interval is required. Both normal recognition gaps and error/reset paths clear the speech latch. Existing stability rules still apply. Voice regression and JS syntax checks passed; refresh loads versioned scripts.

### Next project stage implemented — evaluation mode

Added browser-only trial evaluation (static/evaluation.js and docs/EVALUATION_MODE.md). Refresh, start webcam, select expected letter, record a three-second trial, then download a JSON report before refreshing. One trial counts as one example; frames are correlated. Includes abstention/coverage/per-letter errors and rejection probes; no images are saved or fitted. No automatic model improvements are claimed. Next action is fresh multi-letter testing using this mode and reviewing its report before motion J/Z work. Evaluation metric/cancellation tests, voice tests, JS syntax and Python checks passed; complete interactive workflow still needs a live user check.

### Guided evaluation and simplified reporting

Detailed JSON export was removed at the user's request. The app now tracks coverage across 24 static letters, advances to the next untested letter after a completed trial (without auto-recording), and lists letters needing retests. The HTML report includes coverage/untested letters. User should complete fresh trials and repeat weak/working letters before motion J/Z work. See docs/EVALUATION_MODE.md. Guided progress, report and existing Python tests passed; a live guided session still needs verification. No classifier or cutoff changes were made for this stage.

### Motion-stage foundation added

Separate J/Z/NONE temporal capture, features, training and clip inference are implemented. Read docs/MOTION_STAGE.md. The static model is unchanged. Real motion training data is still pending: there is no production motion.joblib and no J/Z accuracy claim. UI records 2.5 seconds of landmarks locally; interrupted/missing-hand sequences are rejected. Train/validation/test groups and duplicate checks are enforced. Next user action is collect correctly labeled real motion/background sequences across separated signers/sessions, then run scripts/train_motion.py and validate before calling motion recognition complete. Synthetic tests are infrastructure-only. Changes remain local until another publication.

### Public motion-data search

Found SignAlphaSet v3 as a candidate for ASL alphabet motion training. Publisher API verified CC BY 4.0 and ASL_dynamic.zip (~1.25 GB); metadata saved in docs/signalphaset-metadata.json. Actual J/Z videos and signer IDs have not yet been inspected, and no motion model has been trained. Read docs/MOTION_DATA_CANDIDATES.md before downloading/importing; do not infer independent signer splits or reproduce publisher accuracy claims from metadata alone. The user asked about online data to reduce manual training collection.

### Motion bootstrap training completed

User authorized public-data J/Z training while preserving existing features. Selected SignAlphaSet v3 clips were integrity-checked and 76 real sequences extracted (10 J, 10 Z, 56 background); four background failures were logged. Because signer/session metadata is absent, trained a fixed-C bootstrap SVM without claiming held-out accuracy or inventing partitions. models/motion.joblib is now active separately from the unchanged static model; source candidate is models/motion-public-bootstrap.joblib. Read latest docs/MOTION_STAGE.md and reports/upgrades/motion-bootstrap.json. Static classifier, static app script, voice code and A audio preservation checks passed. All 13 Python tests and voice/evaluation checks pass. Next user action: refresh, select Recognize motion and test fresh J/Z/background recordings before fitting them; motion remains experimental. Changes are local, not yet pushed.

### Latest J diagnosis — candidate not activated

Mirroring the saved J clips did not correct the J→Z confusion. User approved two J clips for adaptation; separate models/motion-J-candidate.joblib ranks reserved J examples correctly, but reduces the single working Z clip below the unchanged 70% motion threshold. Active models/motion.joblib was deliberately NOT replaced. Read latest docs/MOTION_STAGE.md and ignored data/processed/motion-J-adaptation.json for exact roles/results. Need more varied Z/background data and fresh reserved recordings before promotion. Do not assume the candidate is active or the J issue is solved. Existing static/voice features are unchanged.

### Latest recorder fix

User could not collect clips because one missed detection immediately cancelled recording. Fixed static/motion.js: tolerate brief missed frames, show valid/missed counts, validate completeness only at recording end. Backend requirements remain eight frames, 0.7-second span, max 750 ms gap. Long loss rejects rather than inventing coordinates; Stop/source/request failure still cancels. Added tests/test_motion_ui.cjs; its gap/Stop checks and 13 Python tests pass. No classifier changes. Refresh before recording more Z/background clips.

### Latest motion model — balanced local adaptation

Active motion.joblib now contains the balanced local J/Z/background candidate. Eight approved selected user clips were added to training; the other eight stayed outside fitting for development comparison (2/2 J, 3/3 Z accepted; 3/3 backgrounds rejected). This is not independently measured accuracy. Read latest docs/MOTION_STAGE.md and ignored data/processed/motion-balanced-adaptation.json. Previous motion weights remain at models/motion-pre-balanced.joblib. Static classifier and voice are unchanged. Fresh later-session J/Z/background evaluation is the next action; do not refit those new evaluation clips before scoring. Local updates not yet pushed.

### Latest live-motion issue and recorder changes

User reports J always uncertain and Z distance/height sensitivity, with too-few usable frames. Improved recorder readiness (wait for first detected hand, then 2.5-second capture), faster scheduling only while motion active, and explicit candidate/score/frame-count diagnostics for abstentions. No new motion-model fitting or cutoff lowering. Refresh needed; backend restarted to expose candidate/threshold fields. Ready-hand, timeout, gap tests and 13 Python checks pass. Ask for fresh clip results including top candidate/score after this recorder update before attributing failures solely to model, pose or distance.

### Latest motion investigation after user frustration

User still reports live J failure despite earlier small development gains. Stopped asking for more recordings. Found timing sensitivity with offline pause stresses; augmented SVM regressed originals and was rejected. Dynamic-time-aligned nearest-template classifier is now active, using unchanged approved training roles. Same development originals 8/8, start holds 8/8 (prior 5/8), end/both holds 7/8. These are NOT independent/live accuracy. No new user data fitted, no threshold lowering, no static/voice changes. Rollback models/motion-pre-aligned.joblib. Read latest docs/MOTION_STAGE.md; live reliability still unresolved until normal use verifies improvement. Fourteen Python checks pass; endpoints smoke-tested with old development clips. Local changes not yet pushed.

### Latest user result — J still rarely works live

User reports Z mostly works but J rarely recognized. Do not claim the aligned model resolved live J. Added an explicit Save last attempt for review button to persist the most recent recognition sequence with the user-selected correct label, validation-only under motion_review. No training/model/voice changes. User can save one failed attempt without re-recording a batch; then inspect that fresh sequence. Existing 14 Python tests and recorder/review JS tests pass. Current task remains diagnosing real live J failures; old development scores are insufficient evidence.

### Latest interface integration — Auto J/Z

User wants J/Z without visiting Motion lab. Added optional Auto J/Z toggle to main webcam, default off. Rolling landmark windows + generic movement gate + two agreeing motion classifications at 70% can overlay J/Z on the main result and reuse voice clips. No weights/voice assets changed. Auto mode pauses during static evaluation/manual motion recording and is webcam-only. Logic/voice/recorder/evaluation tests and 14 Python checks pass; actual rolling detection is experimental and still needs live validation. Read latest docs/MOTION_STAGE.md. Refresh then enable Auto J/Z to use it; manual lab remains for diagnosis.

### Latest Auto Z/P display distinction

User says Auto J now recognizes J but Z appears as P. Motion model cannot emit P; static branch was displaying/announcing P before temporal confirmation. Added a temporary static pause during detected movement and explicit motion candidate/score feedback; normal static resumes after settling. No model weights/cutoff changes or hard-coded P→Z mapping. Refresh loads auto-motion v2 / motion-display v11 scripts. Gate/voice tests and 14 Python checks pass; actual Z reliability remains to be checked in live use.

### Latest E audio and J/Z ending-pose feedback

User reports E sounding like E/I blend and delayed J/Z with static P shown for Z. Regenerated E audio from “e.” and cache-busted voice files; audible confirmation pending. Fixed confirmed motion expiry: J/Z stays while its ending shape is held, releasing after a changed shape/no hand/source reset. No forcing P→Z and no weight/cutoff changes. DTW still requires whole gesture and repeated rolling agreement, so latency remains. New latch/audio mapping tests plus 14 Python checks pass; live recognition accuracy is not proven fixed.


### Completed-gesture Auto J/Z update

Supersedes rolling-window repeated agreement above. Auto J/Z now segments movement and waits for a 450 ms ending hold before classifying one completed sequence. Uncertain/background/error outcomes stay visible while the ending pose is held, preventing static P from replacing unresolved motion. A changed handshape or removing the hand resets the outcome. Incomplete recordings are rejected; motion threshold remains 70%, and model weights/static recognition are unchanged. E uses regenerated Samantha audio (e.) with cache refresh. Fourteen Python tests and completed-gesture/voice JavaScript checks pass. Actual pronunciation and live J/Z reliability still require user verification. Refresh the app, enable Auto J/Z, complete the gesture, and briefly hold still. Changes remain local.


### Simplified main screen
Main screen now keeps camera/media, recognition, text composition and speech. Evaluation, Motion lab and dataset collection are available through Development tools (/?tools=1); their DOM controls remain hidden on the main view to preserve existing initialization. No datasets removed. Full visual redesign awaits user reference choice.


### Main controls refined
Replaced internal model badge with Ready to recognize. Removed motion toggle and acceptance slider from DOM and JavaScript handlers. Webcam motion now runs automatically (still pauses during evaluation/manual collection); static acceptance stays at the requested 30%, motion at 70%. Internal model name remains available to evaluation reports. Recognition reliability remains unresolved. Local-only changes, do not publish without a fresh user request.


### Refero / Dala inspired local frontend
User supplied https://styles.refero.design/style/e5f5f8cf-e68d-4ed1-bbf5-6b67569af648 and style tokens. Replaced stylesheet with black/violet typography-led responsive layout, original decorative hand constellation SVG, and new hero copy. Camera remains primary action; recognition/text/voice and secondary development view preserved. No remote font dependency or proprietary font bundled; Helvetica/Inter fallback stack. Removed technical model/toggle/cutoff controls remain removed. Both template routes, unique IDs, SVG delivery, voice and motion checks pass. Browser visual QA not performed in this session. Local only; do not publish until user explicitly requests. Recognition reliability remains unresolved.


Hand constellation now animates with a nine-second gentle float and four staggered shimmering particle groups. Animation is contained in the decorative SVG, cache-busted, and disabled for prefers-reduced-motion. No recognition changes.


### Browser-verified landing correction and separate signing page
Browser screenshots confirmed prior brain was two ellipses and L/V/Y were sparse polygon outlines. Replaced with denser filled rounded silhouettes and a side-profile brain particle form; original adaptation, not the proprietary reference animation. Landing now at / with only hero.js; Start signing navigates to /sign, which preserves camera, recognition, text, speech and development tools. Legacy /?tools=1 still works. Inspected desktop 1440px and mobile 390px screenshots in headless Chrome. Verified navigation, visible workspace controls, pause holds canvas unchanged, reduced-motion load, no mobile horizontal overflow, no JS errors. Webcam accuracy not validated by these UI checks. Local server restarted; no publication authorized.


### Final reference reproduction and token consolidation
Revisited actual Refero screenshot AND live dala.craftedbygc.com in Chrome. Rejected custom sign morphs removed. Actual reference brain point/color data converted for local canvas animation; provenance and local-only redistribution caveat documented in docs/FRONTEND_REFERENCE.md. All supplied JSON token values read and consolidated into static/tokens.css; malformed verdant hex and CSS range values normalized. Visible yellow eyebrow/links/steps and brain particles, violet CTA, white display typography. Proprietary font not bundled. / landing and /sign workspace retained. Desktop/mobile browser checks pass for navigation, animation pause, reduced-motion load, no JS errors or horizontal overflow; screenshots inspected and desktop heading wrap/mobile particle density corrected. No publication authorized.


### Static recognition regression fixed
Removing the Auto J/Z toggle had enabled motion arbitration for every webcam session. Generic movement could latch uncertain/incomplete motion and suppress static results indefinitely. Main camera now restores static-only recognition, same 30% cutoff and four-frame stabilization; experimental motion remains in Development tools. AUTO_MOTION_ENABLED=false is intentional until motion arbitration is validated. No weights changed.


### Project next stage: measured static evaluation
Frontend polish deferred by user. Existing evaluation start crashed because it referenced removed cutoff DOM control; fixed. Acceptance stays 30%. Next user action is a small A–E trial session at /sign?tools=1 and readable HTML report, followed by 24-letter repeated trials/rejection checks. Evaluation is separate from fitting. Recognition accuracy is not established by browser mocks.


### Message composition stage
User reports J/Z still failing and asks to move forward. Motion remains unresolved and disabled on main webcam. Added Space, Delete last, Clear, Copy text, Speak message and Stop speech to manual composer. Full messages use an available local named female English browser voice; no cloud speech and no male fallback. Live letter speech pauses while a message plays; editing/reset stops message playback. Existing letter WAV playback preserved. This is manual fingerspelling composition, not sign-language sentence translation.


### Report and demonstration milestone
User completed message workflow and requested progress. Added docs/PROJECT_REPORT.md and docs/DEMO_CHECKLIST.md, grounding baseline metrics in saved reports and distinguishing later personalization/current live uncertainty. J/Z explicitly unfinished. README updated for separate landing/signing pages and fixed 30% setting. No new live accuracy claim, no training or publishing. Next: collect quantitative fresh static trials, insert actual results, then prepare presentation if requested.


### Limited phrase stage started
User approved expansion beyond fingerspelling. Added proposed ten-meaning vocabulary, separate phrase data audit script and docs/PHRASE_STAGE.md. Researched official How2Sign (continuous) and ASL Citizen (isolated) sources; target phrase coverage unverified. No phrase model trained, no video download, no existing model changes. Need fluent-reviewed complete phrase data before training.


### Dataset organization audit — 9 November 2026
Moved physical data directory to datasets; preserved data -> datasets and datasets/processed -> preprocessed aliases. No records/images/weights deleted or retrained. New datasets README and DATASET_REPORT document raw sources, preprocessing, exclusions, source licenses, counts, adaptation overrides and unused phrase sources. Audit: 5,200 public static PNG / 4,800 selected / 4,785 features; 71 local static (34 fit,23 diagnostic,14 outside latest ledger);80 public motion /76 extracted;26 local motion (8 fit,8 development,10 later outside ledger). Private inventory hashes 10,701 physical raw/preprocessed files,1,877,738,002 bytes. New paths protected in .gitignore. Run scripts/verify_dataset_inventory.py; docs metadata/baseline reports are snapshots. Model weights remain models/.


### GitHub sync preparation — 12 November 2026
User renewed request to sync. Prior automatic review blocked unlicensed reference-derived brain geometry. Safer publication now excludes static/brain-particles.json (preserved locally, Git ignored). hero.js has an independently authored mathematical particle-orb fallback; no reference geometry embedded. Private recordings, landmarks, inventories and weights remain excluded. Publish reviewed source/docs/tests/assets to MTJ19/Ml-Project-sign-Language- main using a non-force, expected-head update. No claim that J/Z or phrase training is complete.
