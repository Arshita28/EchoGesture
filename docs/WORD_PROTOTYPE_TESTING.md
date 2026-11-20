# Testing the first word prototype

The model is trained for **computer, who, deaf, year and cool**. This is an explicitly declared five-word prototype, not a completed 30-word model.

Open http://127.0.0.1:8081/words while the local server is running. Watch a training example, click Start camera, allow camera access, then click Record 3 seconds and perform the complete ASL sign once. Keep face, shoulders and both hands within view. You can instead upload WebM, MP4 or MOV, at most 10 seconds and 8 MB. Results appear after local processing. Recordings are removed after prediction and never added to training.

Start testing with computer or year. The tiny test subset had one successful computer clip and two successful year clips. These counts do not establish which signs will work for a new person. Cool failed its single test example; deaf succeeded on one of two. Live user testing is still needed.

## Training and measured results

- Training clips: 18. Validation clips: 6. Test clips: 7. Official partitions retained; signer overlap exists.
- Compared linear SVM, two RBF SVM settings and dynamic-time-aligned nearest training templates.
- Selected temporal templates on validation macro F1 only (4/6 validation clips correct). Only that winner was evaluated on test data.
- Test result: **5/7 correct (71.4%)**, macro F1 0.6333. This is a small, quality-filtered available-data subset, not general live accuracy or WLASL100 performance.
- Model artifact: 356,404 bytes, unquantized. No INT8 conversion or mobile latency claim.
- Real video upload through the running API correctly returned computer on an already evaluated test clip in approximately 1.14 seconds on this computer. This is a smoke check, not new accuracy evidence or p95 latency.
- Unknown handling: insufficient hand/body detection is rejected. Other unsupported signs may match one of the five classes. Scores are raw decision ranks, not calibrated probabilities.

Reference examples are taken only from training clips. The page does not collect evaluation recordings, retrain models, or alter the alphabet pipeline.

## Reproduce or restart

From the EchoGesture root with the project dependencies installed:

```sh
python -m scripts.train_words
python -m scripts.predict_words path/to/isolated-sign.mp4
python -c 'from app import create_app; create_app().run(host="127.0.0.1", port=8081, threaded=True, use_reloader=False)'
```

Re-running training overwrites only the word prototype and its report. Preserve this report before changing training or selecting more settings; the seven test clips have now been inspected and cannot support claims of fresh independent tuning.

This checkout currently uses the original project's trusted virtual environment for execution. The same compatible environment and pinned dependencies are needed after restarting. The existing server on port 8080 is the original project; use port 8081 for this word prototype.

## Verification

Thirty Python checks passed, including word inference, invalid uploads, upload cleanup and training-only reference serving. JavaScript checks cover three-second recording submission, oversize handling, camera release and stale-result cancellation. The page was visually inspected in Chrome and its reference video loaded. Automated browser upload was unavailable because the extension lacks file-URL access; the actual upload endpoint was verified separately using a real clip. Actual webcam signing and browser camera permission still require the user's test.


## Independent web evaluation update on 20 November 2026

The frozen model scored **6/10 original unseen web clips**, with two clips per word. Both computer clips became year, and both deaf clips became who. Who, year and the temperature sense of cool each scored 2/2. This small convenience sample does not represent ten independent signers or prove live reliability.

Nine predefined conditions produced 90 correlated trials. Correct predictions out of ten: original 6, dim 5, low resolution 6, smaller signer 6, shifted frame 5, in-plane roll 5, leading hold 5, trailing hold 5, both holds 2. Both holds introduced six detection-quality abstentions. Rejections count as incorrect in these all-attempt rates. This is evidence that both classification generalization and fixed-window coverage need work. Image roll is not a true change of 3D viewing angle.

See `reports/word-external-evaluation.json` for source provenance, exact segments and aggregate results. Clips were checked against known model data hashes and source URLs; identical content re-encoded elsewhere and signer independence are not guaranteed. No weights or thresholds changed during testing. Future evaluation must use new clips if these findings guide development.


## Candidate development update on 20 November 2026

The active prototype remains unchanged. A mask-aware temporal candidate with other-word rejection improves the historical test to 6/7 but reduces known-word validation recognition and scores only 5/10 on the inspected web originals. It is not promoted. Those old clips now serve as development regressions, not independent evidence.

Eight additional official training-only YouTube clips were acquired, excluding the web evaluation URLs. Four pass the existing detection-quality gates, raising known-word training from 18 to 22 clips. The selected expanded template model retains 4/6 validation and 5/7 historical-test performance. With the full prediction quality policy, inspected originals remain 6/10; dim, roll and shift become 6/10; zoom-out becomes 7/10; both-end holds remain 2/10. Additional templates have not solved timing sensitivity. The candidate stays separate in `models/words-expanded-candidate.joblib`.

Reports: `reports/word-masked-candidate.json` and `reports/word-expanded-candidate.json`. Neither candidate is live on the testing page. The next step is dense gesture segmentation and investigation of failed hand coverage on the new deaf/year/cool clips. We will not use a lower detection gate merely to display more guesses.


## Active dense model update on 20 November 2026

Open **http://127.0.0.1:8095/words** for the updated local prototype. Earlier URLs on port 8081 are obsolete. The server now uses `five_word_dense_candidate_v3` and routes inference through dense chronological tracking before visible-region cropping and 32-frame sampling. It remains a single-sign recording/upload workflow.

Training now uses 36 clips, validation 8, and the historical test 8. Selection used validation macro F1 only: 7/8 validation correct (0.8667 macro F1), including 6/6 clips shared with the old validation set versus 4/6 previously. Historical test regression is 7/8. The denominator changed because dense extraction recovers previously excluded clips. The old test has been inspected and is not fresh evidence.

On the already inspected web regression set, the dense candidate gets 9/10 originals and 10/10 with holds at both ends, versus the original model’s 6/10 and 2/10. These are development regressions and do not establish general 90% accuracy. One WHO clip becomes DEAF, and one historical COOL clip still becomes WHO. Unknown or idle gestures may receive a known label. Camera performance remains unmeasured.

The old model is preserved exactly as `models/words-prototype-v1.joblib`; all original alphabet and J/Z weights remain unchanged. `reports/word-model-promotion.json` records the activation. Rollback requires restoring the original model file; inference still supports its sparse preprocessing. New dense model is unquantized and does not meet all hackathon requirements.

New commands: `python -m scripts.extract_word_dense`, `python -m scripts.train_words_dense`, and `python -m scripts.evaluate_dense_candidate`. Training writes the candidate only; activation was a separate verified step. 39 Python tests and recorder JavaScript behavior checks passed.


## Fresh dictionary evaluation on 20 November 2026

A newly reserved dictionary batch contained nine supported-word clips (two each for computer, deaf, cool and who; one year) and four unrelated hello/food clips. These were not fitted. Source basenames and hashes were compared against used datasets and the previous web set; re-encoded copies and signer overlap cannot be ruled out.

The active dense model recognized 4/9 supported originals and incorrectly assigned known words to all 4/4 unrelated signs. Dim versions scored 5/9 supported, and both-end pause versions 4/9. The previously inspected 9/10 result did not generalize to this batch.

A validation-calibrated distance/gap rejection candidate accepted 3/9 originals correctly, abstained on six supported signs and rejected 0/4 unrelated signs (dim: 1/4 rejected). It is not enabled. This batch is now inspected and must be treated as a development regression if reused. Full source and trial records: `reports/word-fresh-dictionary-evaluation.json`.

Next development trains real other-word negatives with the same dense preprocessing; threshold-only rejection is inadequate. The application remains an experimental prototype and the 7/8 validation figure is not a live or fresh-web accuracy claim.


## Other-sign rejection enabled on 20 November 2026

The active local model is now `five_word_other_sign_candidate_v4`, trained using the 36 known-word clips plus 70 other-word training clips. Selection used the original known/negative validation set: 7/8 supported words correct and 5/8 other words rejected, preserving supported validation correctness while improving rejection from zero. No dictionary evaluation clip was fitted.

On the inspected dictionary regression, supported originals remain only **4/9 correct**, while unrelated-sign rejection improves **0/4 to 3/4**. Dim supported clips are 5/9 with 4/4 unrelated signs rejected; pause clips are 4/9 with 2/4 rejected. These are small correlated diagnostic results, not general accuracy or universal unknown detection. One hello example still incorrectly matches computer. The earlier distance-only rejection candidate was rejected because it withheld too many known words without recognizing unrelated signs.

The page now displays no word when the pooled other-sign class wins, and shows an uncertainty message. No continuous live recognition or idle guarantee is implemented. Model backup: `models/words-dense-v3.joblib`; all alphabet weights unchanged. Current address remains http://127.0.0.1:8095/words . Reports: `word-background-training.json`, `word-background-dictionary-regression.json`, `word-background-promotion.json` under reports. Tests: 44 Python checks and recorder UI behavior checks pass.
