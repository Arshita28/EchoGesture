# EchoGesture project memory

Last updated: 16 November 2026. Read this first to resume with minimal context. Read detailed files only when needed. Update this summary after meaningful work; do not copy chat logs, secrets, or private recordings here.

## Goal and decisions

- Build EchoGesture for Samsung XR Hackathon 2026 Track 02 using the existing ASL project.
- Phase 1 is the existing 24 static ASL letters, excluding J and Z. Preserve the baseline and existing functionality.
- First priority is a working recognition pipeline. Website versus installed app is deferred. Do not start a redesign or deployment merely because either option was discussed.
- Extend to word signs using movement across frames. Isolated word recognition is the first milestone; unrestricted sentence translation is not implemented.
- User prefers direct execution and clear goals. A five-word experimental model has now been trained at the user's explicit request; no production accuracy claim. Official WLASL metadata and 167 videos have been acquired. Word features are extracted, but the planned 30-class training dataset remains incomplete.

## Locations and repository

- Active checkout: `/Users/mtj/Documents/ChatGPT/sign Language detection/echoGesture`
- GitHub: https://github.com/MTJ19/EchoGesture (private).
- Branch: `main`, tracking `origin/main`. Last synchronized commit: `3dffa1dd249ef06f383843f826dbe68d04c5f593`.
- Original project: `/Users/mtj/Desktop/ChatGPT/sign Language detection`; left intact. `phase1-source` remote points there.
- Original GitHub: https://github.com/MTJ19/Ml-Project-sign-Language-
- Original tracked source/history imported into EchoGesture, followed by README, continuation notes and roadmap additions. GitHub and local checkout were verified to have identical commit and tree.
- Local datasets and model checkpoints were copied; they remain Git ignored. Virtual environment was not copied. Reference-derived `static/brain-particles.json` was omitted; the original fallback is available.
- CLI Git lacks saved GitHub credentials. Connected GitHub tools work. Use them when needed; never assume CLI push authentication. Remote commit objects were reconstructed locally from verified GitHub metadata to synchronize the checkout.

## Existing implementation

- Flask/browser interface with separate landing page `/` and signing workspace `/sign`.
- Pipeline: frames -> MediaPipe Hands -> normalized landmark features -> SVM -> acceptance and stability -> displayed letter.
- Static acceptance is 30% with four-frame stability. This is a user-selected operating setting, not a validated optimum.
- Manual message composition and speech exist. Experimental J/Z tools exist but remain unreliable and disabled on the main webcam path. Phrase groundwork exists, with no trained phrase model.
- Original static pilot: 90.98% accuracy on 1,198 extracted images from one held-out signer, not general webcam accuracy. Later personalized model has no fresh independent accuracy claim.
- Do not retrain on diagnostic/test captures, overwrite baseline reports, or claim J/Z or phrase recognition is complete.

## Development phases

1. Preserve and reproduce the 24-letter baseline.
2. Audit word-sign data and prepare labeled video sequences.
3. Train and evaluate a compact temporal word classifier; expose `predict(video_path)`.
4. Integrate live captions, idle/uncertain handling and duplicate suppression; retain alphabet mode.
5. Quantize and measure local edge inference; choose interface after compatibility checks.
6. Prepare reproducible source, predictions, results, model conversion and demo.

## Dataset plan

- Retain ASLYset for Phase 1. The earlier project also includes SignAlphaSet motion experiments; see existing docs before reuse.
- Primary next dataset: WLASL, individual ASL word videos. Official source: https://github.com/dxli94/WLASL
- Audit WLASL100 availability; first prove 20–30 well-represented classes, then expand toward 100 if feasible. These are proposed development milestones, not organizer-approved minimums.
- Preserve official train/validation/test assignments; audit signer overlap, class balance, missing videos and duplicates. Never split frames from the same clip across partitions.
- Candidate approach: both-hand and upper-body landmark sequences with a small temporal convolutional model or GRU. Architecture remains undecided until baseline review and data inspection.
- Collect separate, correctly performed real-camera evaluation clips and idle/background examples. Keep evaluation separate from fitting.
- How2Sign is a later option for continuous signing and translation: https://how2sign.github.io/ . Confirm organizer requirements before excluding it from final scope.

## Hackathon constraints and unresolved questions

- Fully local visual processing, zero cloud video streaming; exocentric signer view; captions or audio; quantized model and reproducible clip prediction.
- Deck targets: INT8 model under 20 MB, p95 latency at most 50 ms, Top-1 accuracy at least 70%. Measure on the actual device; report full pipeline costs separately from classifier time and sign observation delay.
- Results template additionally lists robustness drop at most 5 percentage points, average power at most 500 mW and 10-minute thermal drift at most 20%. Test definitions require clarification.
- Judging: quantization 20%, latency 20%, Top-1 accuracy 35%, robustness and HUD 25%.
- Deliverables: repo and README, `predict()` accepting a clip path, predictions, results.csv, quantized model and conversion script, demo and presentation.
- Deck conflicts: 20 versus 25 MB; predictions.jsonl versus predictions.json; 18 versus 22 November deadline; 3 versus 5 minute demo. Aim under 20 MB and at most 3 minutes; verify deadline and schema with the latest organizer instructions.
- Confirm required vocabulary, isolated versus continuous evaluation, scope of latency/model-size accounting and whether both datasets are mandatory. Do not present our small vocabulary as full compliance without clarification.

## Files to consult selectively

- `docs/EchoGesture_Development_Roadmap.docx`: three-page phased plan with completion checks.
- `docs/ECHOGESTURE_START.md`: migration and continuation notes.
- `PROJECT_HANDOFF.md`: detailed historical implementation and experiment ledger; old repository and paused-work references are historical.
- `README.md`: run and baseline reproduction instructions.
- `docs/UPGRADE_LOG.md`, `docs/MOTION_STAGE.md`, `docs/PHRASE_STAGE.md`: current experimental limitations.
- `datasets/README.md`, `datasets/DATASET_REPORT.md`, `reports/RESULTS.md`: data provenance and baseline evidence.
- Hackathon source: `/Users/mtj/Documents/XR_Hackathon_2026_Consolidated_Slides_V3-2.key`.

## Latest implementation milestone

- Existing 14 Python tests passed before changes. Added four WLASL audit tests; all 18 pass. All four JavaScript test scripts pass.
- `python -m scripts.verify_phase1_baseline` reproduces validation macro F1 0.8886399944 and test accuracy 0.9098497496 with the preserved baseline. No refitting; input feature hash verified. Report: `reports/phase1-reproduction.json`.
- All 16 model checkpoint hashes match the original project.
- Official WLASL v0.3 metadata: 2000 classes and 21083 listed clips. First 30 classes: 709 clips (497/117/95 train/validation/test), 47 signer IDs across official partitions. WLASL100: 2038 clips (1442/338/258), 72 signer IDs across partitions. Do not claim signer-disjoint WLASL splits.
- `scripts/audit_wlasl.py` validates metadata, preserves official splits and IDs, audits local file presence and duplicate hashes, and writes manifests. It never downloads videos, decodes or trains. Training readiness stays false until later validation. Metadata and manifests are Git ignored; aggregate source hash/counts are in `reports/wlasl-metadata-audit.json`.
- See `docs/WORD_RECOGNITION_STAGE.md` for commands, reusable components and next steps. Current single-hand extractor must remain unchanged for alphabet mode; implement a separate word feature pipeline.
- Tests use the original Desktop project's trusted Python virtual environment. EchoGesture still needs its own environment before independent use.

## Latest acquisition and extraction milestone

- `scripts/acquire_wlasl.py`: bounded official direct HTTPS MP4 downloads with complete decode checks; no TLS bypass, mirrors or automatic YouTube/SWF handling. All 345 eligible entries attempted. 167 valid local clips, 178 failed (82 HTTP 404, 62 DNS failure, 34 certificate hostname mismatch). Skipped 133 HTTP/other scheme, 229 YouTube, 1 trim-required and 1 format-conversion entry.
- Official preprocessing policy reviewed: direct isolated clips copied whole; only acquired entries with start 1/end -1. Do not trim them again. Terms retained at ignored `datasets/raw/wlasl/C-UDA-1.0.pdf`.
- `asl_lab/word_features.py`: separate Holistic extractor, 154 float32 features with actor-relative hand slots, wrist shapes, shoulder-relative wrist/body positions, visibility and masks. Pseudo-3D, not calibrated metric 3D. No face features. Original alphabet extractor unchanged.
- `python -m scripts.extract_words METADATA`: 32 uniformly sampled positions per clip, model reset between clips, no filled-in missing landmarks. All 167 clips extracted; 118 pass provisional body >=70% and any-hand >=50% coverage gates; 49 fail. Cached-report JSON serialization fixed and regression-tested.
- `python -m scripts.audit_word_sequences`: verifies labels, feature shape, source hash, timestamps, masks, duplicate content and partitions. No integrity errors. Only computer, who, deaf, year and cool have quality-passing examples in all three partitions. Do not silently train a reduced vocabulary or call this a complete 30-word dataset.
- 27 Python tests pass. All 16 original model checkpoints verified unchanged. No word model trained or UI integration performed.
- Aggregate report: `reports/word-data-progress.json`. Local detailed records: `datasets/raw/wlasl/acquisition.json`, `datasets/preprocessed/words/extraction.json`, `readiness.json`, `quality-manifest.json`; media/features remain ignored.
- MediaPipe real extraction ran successfully with macOS graphics access outside the sandbox. Full desktop Holistic size/latency are not validated for edge deployment.

## Latest word training and testing milestone

- User explicitly requested model training and testing a few words. Declared `config/word_prototype.json`: computer, who, deaf, year, cool. The incomplete 30-class scope is not silently relabeled as complete.
- `python -m scripts.train_words`: 18 training / 6 validation / 7 test quality-passing clips, original splits. Four candidates fitted only on training. Dynamic-time-aligned template classifier selected using validation macro F1 (4/6 correct). Selected winner scored 5/7 test clips correct, macro F1 0.6333. Cool missed its single example, deaf 1/2; computer 1/1, year 2/2, who 1/1. Too small and signer-overlapping for live accuracy claims. Do not tune against these inspected test clips.
- New artifact `models/words-prototype.joblib` (356404 bytes) is separate from all 16 original checkpoints. Unquantized, no edge compliance claim. Full results: `reports/word-prototype-training.json`.
- New inference: `asl_lab/word_model.py`, `word_inference.py`, `word_routes.py`; CLI `python -m scripts.predict_words VIDEO` exposes video -> JSON. Quality rejection only; unknown/background rejection is not trained. UI calls results closest supported words, never calibrated confidence.
- New `/words` page: reference videos from training only, 3-second browser recording or <=10-second/8-MB upload. Uploads processed locally in temporary directories then removed; no training capture. Existing alphabet remains at `/sign`.
- Local EchoGesture server running at http://127.0.0.1:8081/words in exec session 36746 using original trusted Desktop venv, with macOS graphics access. Original port 8080 unchanged. If process is gone, restart using docs/WORD_PROTOTYPE_TESTING.md.
- Real upload API smoke returned computer correctly in ~1.14 s for previously evaluated clip. Not a new benchmark or p95 result.
- 30 Python tests passed. Word JS behavior checks pass. Chrome page and reference video visually verified. Browser automation file upload lacked extension file-URL permission; actual API upload verified with curl. Do not ask user to change extension settings; manual camera/upload testing is the next user action.
- See `docs/WORD_PROTOTYPE_TESTING.md`. No user webcam result yet.

## Independent web evaluation on 20 November 2026

- User reports deaf->who, inconsistent year and cool failures, and asks us to test unseen web clips instead of relying on their signing. They do not know ASL. Do not request repeated manual signing as the primary evaluation method.
- Acquired ten new public dictionary videos (two per word). Longer compilations trimmed using label-specific WLASL 25-FPS annotations, converted to seconds against native video FPS. Sources/segments retained under ignored datasets/raw/web-evaluation. Never train on this evaluation set without explicitly reclassifying it as development data.
- Frozen model evaluated under nine predefined conditions, 90 correlated trials. All original and transformed clips processed. No fitting, model replacement or threshold tuning. Hash verified unchanged.
- Original 6/10 correct: computer 0/2 (both year), deaf 0/2 (both who), who 2/2, year 2/2, cool 2/2. Tiny convenience sample, not ten distinct signers. Cool samples use temperature sense; do not claim all cool variants solved.
- Dim 5/10, low-res 6/10, zoom-out 6/10, shifted 5/10, in-plane roll 5/10. Leading or trailing one-second holds each 5/10. Both holds 2/10 with six detection abstentions. All-attempt accuracy counts abstentions as incorrect; transformed versions are not independent samples.
- Conclusion: original classification errors exist even with adequate detector coverage; pauses also break fixed-window detection coverage. Live mode needs gesture segmentation, background rejection and stronger training diversity. Do not add continuous guesses and call reliability fixed.
- Tooling: scripts/prepare_external_segments.py, scripts/evaluate_words_external.py, config/word_external_evaluation.json. Tests: 32 Python checks pass. Report: reports/word-external-evaluation.json. Detailed trials/features remain ignored.

## Candidate experiments on 20 November 2026

- Added `asl_lab/word_candidate.py`: mask-aware temporal distance with body-relative wrist location; outer missing-frame trimming preserves internal gaps. Three location weights compared on original training/validation, other labeled training words used as NONE templates. No production model replacement.
- `scripts/develop_word_candidate.py` selects weight 1 on mixed known/other-word validation: 3/6 known correct, 6/8 other words rejected. Original historical test 6/7; inspected web original 5/10 and both-holds 4/10. This is a rejection/known-word tradeoff, not a deployment improvement. Saved models/words-masked-candidate.joblib and reports/word-masked-candidate.json. Those diagnostic paths evaluate candidate behavior, not original production quality policy.
- `scripts/acquire_word_training.py` attempted 12 official train-only YouTube entries, excluded all known web evaluation URLs and original extracted video IDs. Eight downloaded and decoded; separate output datasets/raw/word-training-extension. Four pass unchanged hand/body coverage gates (new computer and who examples); deaf/year/cool additions failed coverage. Existing evaluation sources never fitted.
- Used existing extractor on the eight new clips, outputs datasets/preprocessed/word-training-extension. No official partitions altered.
- `scripts/train_word_extension.py` trains three expanded candidates, selects on six original known-word validation clips only. Expanded original template model selected at 4/6 validation, same macro F1 0.6333; 22 known training clips. Historical test remains 5/7. Saved models/words-expanded-candidate.joblib and reports/word-expanded-candidate.json, not activated.
- Expanded candidate diagnostic comparisons use actual classify_word quality rejection: original 6/10, dim 6/10, low-res 6/10, roll 6/10, shift 6/10, zoom-out 7/10, leading/trailing hold 5/10, both holds 2/10. These are inspected development regressions, not fresh accuracy. Raw forced-choice scores overstate timing improvements if quality abstentions are omitted; report uses full policy.
- All 35 Python tests passed before final report-only/diagnostic policy correction; targeted tests rerun after. Original production word hash and all 16 alphabet/motion checkpoints verified unchanged. No live integration or promotion. No user signing requested.

## Dense model training and activation on 20 November 2026

- User requested further training for accuracy. Expanded official training-only acquisition to all 23 eligible entries: 14 acquired, 9 failed. Evaluation URLs excluded; never fitted web clips. Output datasets/raw/word-training-extension/acquisition.json.
- Added dense preprocessing `asl_lab/word_dense.py`: chronological tracking up to 25 FPS, trim outer frames without visible hand/body before uniform 32-frame resampling, keep internal gaps and masks. This is visible-region cropping, not full learned gesture segmentation; valid held poses and unsupported signs remain limitations.
- `scripts/extract_word_dense.py` created separate features for all 52 target-word local clips; 52 passed cropped coverage. Changed preprocessing/denominators mean this is not directly comparable to sparse extraction yield. Outputs datasets/preprocessed/word-dense.
- `scripts/train_words_dense.py` trained five candidates on 36 training clips only; validation 8, historical test 8. Original template matching selected on validation macro F1 0.8667 (7/8). Shared original validation improves 4/6 -> 6/6. Historical regression 7/8; cool 13198 still -> who. `asl_lab/word_summary.py` provides tested temporal summaries, but not selected.
- `scripts/evaluate_dense_candidate.py`: already inspected web development regression, not fresh test; original 9/10 (who A5dihMAjeTE -> deaf), both-end one-second holds 10/10 with zero abstentions. No web fitting or parameter tuning. Reports word-dense-training.json and word-dense-regression.json.
- Enabled dense candidate for local testing because validation improved and regressions passed. Active models/words-prototype.joblib name five_word_dense_candidate_v3, SHA 8892dcbf0eea3670920e602b1d09510cf9a7d2433b70f9b1ab387004e1931cff, 711738 bytes. Old exact artifact saved models/words-prototype-v1.joblib; candidate also preserved separately. Promotion rationale reports/word-model-promotion.json. No INT8 or edge compliance claim.
- word_inference.py dispatches using bundle preprocessing version; legacy sparse models still supported; unknown versions rejected. UI updated to state 7/8 validation, not claim real-world accuracy. Still isolated recording/upload; no continuous live model or background rejection.
- Local server restarted at http://127.0.0.1:8095/words, exec session 94260. Ports 8081/8082 belong to other services; do not stop them. Original alphabet models untouched.
- 39 Python tests and word recorder JS checks passed. API upload smoke on previously confused deaf clip verifies actual dense path. This is development verification, not new independent accuracy.

## Fresh dictionary and other-sign model update on 20 November 2026

- User requested web tests and continued development; later said continue while this work ran. Do not ask them to learn or perform ASL as primary testing.
- `scripts/acquire_fresh_dictionary.py` reserved 13 dictionary videos: nine supported (computer 2, deaf 2, year 1, cool 2, who 2) plus hello/food four unsupported probes. Excluded known source basenames/hashes, then explicit new-provider WHO links. Re-encoded duplicates and signer independence cannot be guaranteed. Sources under ignored datasets/raw/fresh-dictionary/sources.json, never fitted.
- Frozen dense v3 on new originals: 4/9 known correct, 0/4 unrelated rejected. Dim known 5/9, holds 4/9. This contradicts any general 90% claim from the older inspected batch. Full initial report reports/word-fresh-dictionary-evaluation.json.
- `scripts/calibrate_word_rejection.py` calibrated distance .18 and gap .005 on original eight known/eight other-word validation before fresh inference. Candidate gets only 3/9 supported originals and 0/4 unsupported rejection; not activated. No fresh-web threshold tuning.
- `scripts/extract_word_negatives.py` dense-extracted 78 existing non-target train/validation clips; all passed, split 70 train/8 validation. No web test media read. Files datasets/preprocessed/word-negative-dense.
- `scripts/train_word_background.py`: five six-class candidates, fitting 36 known plus 70 other-word training clips. Pooled NONE class. Selected dense templates on validation balanced known accuracy + negative rejection with known >=6/8 and negative >=4/8 floor. Selected preserves 7/8 known validation, rejects 5/8 other-word validation (old 0/8). Model report reports/word-background-training.json.
- Compared selected model on now-inspected dictionary regression, never fitted: originals 4/9 known correct, 3/4 unsupported rejected; dim 5/9 and 4/4; both holds 4/9 and 2/4. Same known correctness, fewer false accepts. Report reports/word-background-dictionary-regression.json. All these clips now inspected; do not call repeated results fresh tests.
- Activated candidate for local experimental testing. Active model name five_word_other_sign_candidate_v4, SHA 7d4b102e4c6c2d9e5dc1b1138c4cf723e1fc5729ed13bd663de1d3792f81a2b9. Model ~2.09 MB unquantized. Prior dense v3 saved models/words-dense-v3.joblib; first model still words-prototype-v1.joblib. Promotion reports/word-background-promotion.json. No claim supported-word accuracy solved.
- `classify_word` maps NONE to uncertain_word, null label, no candidates. Distance/gap policy also supported for experimental calibrated models. UI handles uncertain_word without displaying null or a guessed word; text now states 4/9 and 3/4 latest dictionary results. Backend/API smoke verifies unsupported food rejection and supported computer path. No continuous live mode or idle guarantee.
- Local server restarted at http://127.0.0.1:8095/words, restarted after hiding internal NONE alternatives; latest server session is recorded in this chat. Do not stop other applications on 8081/8082. All 16 alphabet/motion checkpoints unchanged. 44 Python tests and recorder JS checks pass.
- `scripts/acquire_word_training.py` now excludes fresh dictionary URLs as well as initial external evaluation URLs. Evaluation clips must remain excluded from training unless role explicitly changed and new test set reserved.

## ChatGPT model clarification and comparison work

- User said “use whichever model” meaning the ChatGPT model doing development, not permission specifically to change recognition architecture. Acknowledge this clarification; do not claim ability to switch own underlying chat model. Goal remains continued development and high recognition accuracy, with memory used for concise resumption.
- Before clarification, compared recognition candidates without activating any: scripts/compare_word_ensembles.py (extra trees, random forest, PCA/SVM, regularized LDA) and scripts/compare_word_invariance.py (velocity and handedness-aware temporal matching). Existing 36 known +70 negative train clips; original 8 known +8 negative validation. Ensembles additionally used three-fold signer-grouped training CV; labels missing from folds reported.
- Neither comparison yielded a candidate preserving both active known correctness 7/8 and negative rejection 5/8. Reports word-ensemble-comparison.json and word-invariant-comparison.json; candidate artifacts are separate. No production replacement or new accuracy improvement claimed. More expressive models did not solve sparse-data limitations.
- Added tested estimator adapter asl_lab/word_ensemble.py and experimental asl_lab/word_invariant.py. No current inference change. All 47 Python checks passed. Active model remains five_word_other_sign_candidate_v4; latest known-word dictionary performance still 4/9.

## Next action

Supported-word accuracy on latest dictionary set is still only 4/9. Expand diverse legitimate train/development examples and review sign variants/senses before more architecture tuning. Preserve both inspected web batches as regressions and reserve genuinely new videos for claims. Add real idle/background recordings and semantic gesture boundaries before continuous live recognition; pooled other-word NONE is not universal open-set detection. Keep local page experimental and report wrong/uncertain predictions honestly.

## Current change

Memory, baseline reproduction, WLASL acquisition/audit, word extraction/readiness tooling, tests and reports are local changes, not yet committed or pushed. Last synchronized commit above remains accurate. Update after publication.

Word data audit: supported training clips/distinct metadata signer IDs: computer 11/6, who 8/6, deaf 8/7, year 4/4, cool 5/3. Dictionary mirror provider+filename matches WLASL IDs 13217, 65445, 66816, 69281, including both cool regression clips. None of these IDs occurs in the active known-word training manifest. Training extension acquisition now reserves these IDs even across different hostnames. This check does not detect renamed or re-encoded copies generally. One trained cool filename explicitly says Cool Down; unavailable Handspeak variants mention neat. Review senses before adding clips. No retrain or new performance estimate this pass.

Cool-neat experiment: scripts/compare_cool_sense.py added mirrored Handspeak WLASL training IDs 13205 and 13206 (media.signbsl.com/videos/asl/handspeak/mp4/cool-neat2.mp4 and cool-neat.mp4). Reserved Ziggy ASL cool (awesome) page video as evaluation only; keep its source and report hash excluded from future fitting. Both models predicted who with full sampled hand/body detection. Known validation fell 7/8 to 6/8, other rejection stayed 5/8. Candidate saved separately and declined. Results: reports/word-cool-neat.json. Ziggy clip now inspected.

Wrist diagnostic on inspected dictionary cool/who originals and Ziggy: scripts/diagnose_word_wrists.py, reports/word-wrist-diagnostic.json. Swept hand-wrist weights 0/2(active)/4 while retaining pose features including pose wrists. No setting recognized any of 3 cool; doubling lost the 1 correct who of 2. Ziggy median most-visible wrist-to-nose distance 0.98 shoulder widths versus 1.04 on one who. Medians cover full visible clip, not peak articulation; no landmark ground truth. Active unchanged.

Visual diagnostic: datasets/raw/cool-sense/cool-who-landmarks.jpg shows seven evenly spaced frames of Ziggy cool and dictionary who-4313b25e08c7 with right thumb/index/other fingertips. Cool hand stays near upper chest from ~0.29 s onward; who reaches mouth/chin around 2.13–2.83 s. Visible location distinction survives tracking in these sampled frames. Cannot establish fine finger tracking accuracy or causal attribution from two clips. No model change.

Direct fingertip-location experiment: scripts/compare_fingertip_location.py adds 20 distances from both hands' five tips to pose mouth center and approximate upper chest in shoulder-width units. Same 106 training templates; fixed unit weight. Validation both 7/8 known and 5/8 unrelated rejected. Inspected dictionary + Ziggy both 4/10 supported, 3/4 unrelated rejected. One cool→who and one who→year became abstentions; Ziggy still who. No promotion. Per-clip results reports/word-fingertip-location.json.
Nine-word signer-disjoint benchmark now inspected: 37 train, 21 development, 13 test. Templates 12/21 dev vs pooled linear 3/21; 7/13 test. Separate INT8 template storage with floating-point scoring. See reports/word-vocabulary-benchmark.json. Five-word UI active.

Camera clarification: /words starts preview only; Record 3 seconds triggers inference. User expected automatic detection. Updated initial and camera-ready instructions plus JS cache version. Continuous live mode remains pending. Accuracy explanation: 36 supported training clips, limited sign variants/signer coverage and prototype nearest-template classifier; web 4/10. Do not present UI clarification as accuracy improvement.

Revisited supplied Downloads PPTX V3-2 (1), slide 11 explicitly says predict(video_path)->dict returns predicted gloss/word for one video clip; uppercase prediction from fixed vocabulary, clip_id exact test manifest, optional top5, confidence, latency_ms. One JSONL per test clip. Earlier claim deck unclear on word-vs-sentence was incorrect. User independently confirms words. Prioritize exact organizer vocabulary and manifest; no need sentence translation for scored output. Slide9 smartphone local live demo, slide10 quantized TFLite/ONNX <20MB, p95<=50ms and >=70% sheet slide11. Sheet model_size<=25 conflicting with <20 elsewhere; use <20. Need actual vocabulary file and starter baseline to align training. How2Sign optional secondary source, WLASL appropriate word task.

User clarified vocabulary hidden; pursue broad WLASL2000 coverage. Downloaded official Pose-TGCN checkpoints from author-linked Drive archive 1dzvocsaylRsjqaY4r_lyRihPZn0I6AA_ into ignored datasets/raw/wlasl-official-baseline/pose-tgcn-checkpoints.zip (~52MB). Contains asl100/300/1000/2000 checkpoints, WLASL2000 checkpoint 26,491,626 bytes. Need source architecture, official keypoints/splits and evaluation before any accuracy claim. Source keypoints Drive ID 1k5mfrc2g4ZEzzNjW6CEVjLvNTZcmPanB. Checkpoint not evaluated or integrated. Dataset precomputed pose must match live extractor for eventual deployment.
