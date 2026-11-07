# Training and evaluation protocol

Fit a StandardScaler inside SVM and KNN pipelines on training data only. Random Forest uses the normalized coordinates directly. Baselines: RBF SVM C=10, 200-tree balanced Random Forest, distance-weighted KNN k≤5. Random seed 42. Choose by validation macro F1; do not choose based on test scores. The score threshold 0.70 is fixed before evaluating test. If tuning it later, use validation data and preserve a fresh test set.

`reports/evaluation.json` includes per-model accuracy, macro F1, classification report, confusion matrix label ordering, accepted coverage and accuracy, groups, counts, feature version and data hash. Report both overall and accepted accuracy so abstention cannot disguise poor performance. The saved model is trained only on the training partition used in the comparison; test data is never refitted.

SVM probability estimation uses its internal training-only folds, which are not group-aware. Treat all scores as heuristic confidence values, not calibrated probabilities. For a formal calibration study use signer-group-aware out-of-fold calibration and additional signers. A high score does not prove an unseen gesture is in the alphabet.

Required real-world checks: new consenting signers, new recording day, left/right hands, mirrored inputs, varied lighting/background, partial occlusion, empty scene, two hands, out-of-vocabulary hand poses and movement. Record false accepts and failures as well as successes. Public four-signer pilot scores cannot substitute for this study.

Synthetic test records in `tests/` exercise infrastructure only. They must never appear as ASL recognition results. A blank-frame benchmark measures local no-hand processing latency, not classifier accuracy or complete webcam FPS. Browser polling adds scheduling and encoding time; report full end-to-end latency separately during a live demo.

The user later requested an adjustable browser acceptance cutoff, initially 30%. This changes live acceptance only; `evaluation.json` and the saved model's default remain at the original 70%. No accuracy or accepted-accuracy result at 30% is claimed. No-hand/multiple-hand rejection and four-frame stability remain active. A model score is not signing correctness or calibrated accuracy.
