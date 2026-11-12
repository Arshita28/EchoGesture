# Guided webcam evaluation

Open http://127.0.0.1:8080/sign?tools=1, start the webcam, choose the expected letter, make the sign, and click **Record 3-second trial**. After completion, the selector advances to the next untested static letter; it never starts recording automatically. Check the expected letter before each new trial. You can manually select a letter to repeat it or choose **No sign / rejection check** for empty scenes and unsupported gestures.

Alphabet coverage shows how many of the 24 static letters have been tested and how many have at least three trials. A separate list identifies letters with wrong or uncertain trials. These are priorities for diagnosis, not automatic instructions to retrain or lower the cutoff. J/Z remain deferred. Repeat trials in fresh poses/recording sessions rather than holding one pose repeatedly. Check previously working letters too.

**Download readable report** saves a standalone HTML summary and tables. Detailed JSON export has been removed at the user's request. Export before refreshing: session results live only in browser memory. The report includes alphabet coverage and untested letters so one failed trial is not mistaken for a full alphabet accuracy assessment.

Each trial aggregates sampled stabilized displays. A letter must occupy more than half of responses to be accepted; otherwise the trial abstains. At least five responses are required. One trial counts as one test; correlated frames are not separate tests. Stop/source changes or request errors cancel the active trial. The expected-letter control is locked during recording. Static acceptance is fixed at 30%; the removed slider is no longer required.

Metrics separate wrong predictions, uncertainty, acceptance coverage, accuracy among accepted letters, and false accepts on rejection probes. Median available server time excludes browser encoding, network and polling, and omits responses without timing. Keep the same model and cutoff during a session, refreshing after a model change so its displayed name is current.

Evaluation does not save images, modify training data, upload reports externally or claim signer independence. This user has supplied adaptation training photos; fresh recordings test session behavior, not unseen-signer generalization. New consenting signers are required for that stronger claim. Do not retrain on a trial and continue using it as an independent test.

Checks: node tests/test_evaluation.cjs covers trial outcomes, coverage, rejection probes, cancellation, readable rendering and guided next-letter behavior. Fixtures are logic checks, not measured sign-recognition results. A live guided session still needs user verification.

## Next checkpoint
Start with one three-second trial for each of A, B, C, D and E to verify the workflow. Download the readable report before refreshing. Then test the remaining static letters (all except J/Z), followed by three fresh trials for every letter across varied positions/sessions and five no-sign checks. Move your hand away and reset between trials. Report incorrect letters separately from uncertain results. Use new consenting signers later to evaluate generalization. Do not use evaluation trials for training before recording baseline results.

After baseline: diagnose the largest confusions, adapt only a designated training subset, compare against the reserved evaluation set, then return to J/Z integration. Frontend polish and final project report follow measured reliability.
