# Geometry upgrade — 8 November 2026

User reports live V/C errors and B/I/F, M/S/U confusion. Added all 210 pairwise landmark distances and 15 finger-joint cosines alongside the original 63 coordinates (288 total features). A deterministic SVM grid was fitted only on the original training signers and selected only on validation macro F1. No local webcam photos were used in fitting, and the original test report was not recomputed or overwritten.

Selected geometry SVM C=100, gamma=scale. Validation macro F1 improved from 0.8886 to 0.9430. Validation recalls: C 0.84→1.00, M 0.957→0.978, S 0.90→0.96; U regressed 0.98→0.94; V/B/I/F remained 1.00. This is selection performance, not fresh independent test accuracy or proof of live fixes. The original test signer has previously been inspected; new generalization claims require fresh signers/sessions.

On the seven existing local photos (four B, three C), the candidate ranked all labels correctly; the baseline confused all three C with O. Candidate C scores remained below the 30% live cutoff, so correct top ranking does not yet mean accepted recognition. Local detailed diagnostics remain ignored under data/processed. Further user captures of V/C/B/I/F/M/S/U are pending and must remain outside fitting for this comparison. Capture manifest split labels alone do not make these photos independent signer evidence.

The candidate is active locally; baseline weights were preserved at models/baseline-aspect-v2.joblib. The 30% live default and original 70% baseline reports are unchanged. Refresh the app to see the new model badge. The server reloads the trusted local artifact on its next request. To restore the prior model, copy the preserved baseline weights over models/classifier.joblib.

Reproduce selection with `python scripts/compare_geometry.py`; this saves a separate candidate and validation report without activating it. Original evaluation reports are preserved. Next: inspect new labeled webcam failures, especially U regression and C abstentions, then validate improvements on fresh diverse recordings rather than lowering the cutoff to force labels.

## M/S follow-up

New user-labeled M/S diagnostic captures reproduced M errors. A geometry-only closed-fist specialist trained exclusively on public data was tested; validation macro F1 fell to approximately 0.926–0.930 and the local M failures remained. The specialist was not activated. The current live geometry model is unchanged. Detailed local diagnostic outputs remain ignored. A examples have not yet been captured. Adaptation training would require explicitly reassigning selected diagnostic examples to training and collecting separate fresh evaluation recordings; this would be personalization/session evidence, not independent signer accuracy.

## Consented local A/M/S adaptation

User approved adaptation after the diagnostic-only stage and saved A captures. Three detected frames per A/M/S (nine total) were explicitly reassigned to adaptation training; their roles are recorded locally in data/processed/local-adaptation.json. Original capture manifest labels were preserved. These samples are no longer evaluation data. Other successfully extracted captures were reserved only as same-session diagnostics, not independent validation.

The adaptation candidate is active locally, with the prior geometry model preserved at models/geometry-pre-adaptation.joblib. Public validation macro F1 was 0.9452 (C recall remained 1.00). Among eight correlated reserved webcam frames, six ranked correctly: both A, three of four M, one of two S. One M→S and one S→M error remain. There is no fresh-session or independent-signer accuracy claim. Original public test report and the 30% UI cutoff remain unchanged. Next action is a fresh recording of A/M/S plus a C regression check, kept entirely outside training. Use scripts/adapt_local.py only for an explicit retraining experiment; it does not activate its candidate automatically.

## Full alphabet report and D/P adaptation

User completed a qualitative alphabet test; see docs/ALPHABET_TEST_LOG.md. Saved D/P examples reproduced all five D→X errors and P→H/low-score-P behavior. Extended the consented local adaptation experiment with three D and three P training examples, retaining all earlier A/M/S training examples. Remaining frames are same-session diagnostics, not independent test data. Public validation macro F1 is 0.9460. Both reserved D frames now rank D first (scores about 0.76–0.87). All three reserved P frames rank P first, but two remain below the 30% live cutoff (scores about 0.28–0.30). Existing M/S diagnostic errors remain; do not claim full alphabet reliability.

The DP candidate is active locally, prior weights saved at models/pre-dp-adaptation.joblib. Cutoff unchanged. E/Q/U/X/G and uncertain C captures are pending. These failures have not been fixed merely by addressing D/P. Fresh recording/session checks and working-letter regressions remain required. Detailed photos, features, roles and per-frame reports stay local under ignored data paths.

## Remaining-letter capture batch

New E/Q/U/X/G/C captures were supplied. Selected examples were reassigned to adaptation training alongside earlier A/M/S/D/P examples; at least one detected unique image per class was reserved as correlated same-session diagnostics. Byte-identical captures are deduplicated before splitting. Two P-labeled captures in the for_E group were excluded because their label/group assignment was inconsistent with the requested capture batch, rather than silently relabeled.

The full adaptation candidate is active locally, prior DP weights preserved at models/pre-full-adaptation.joblib. Public validation macro F1 is 0.9491. All reserved examples for E/Q/U/X/C/G ranked correctly above the existing 30% cutoff. P still has below-cutoff cases, and the older M/S diagnostic errors remain. This is not an independent webcam accuracy result; fresh later recordings of both weak and previously working letters are the next required validation. Cutoff and original baseline test reports are unchanged. Photos, detailed sample roles and diagnostics remain local under ignored data/processed/local-adaptation.json. Nine automated checks pass.

## A audio and I/Y follow-up

User reported displayed A sounded like I. Regenerated only the Samantha A clip with the phonetic input “eigh” instead of ambiguous “ay”; incremented audio/script cache versions and matched the fallback spelling. This is an audio change, not an A-classifier change. User audible confirmation is still required. User also reports I→Y when the thumb moves slightly outward. No local I/Y captures are currently saved, so no classifier relabeling or hard-coded thumb rule was introduced. Collect correctly labeled I with small natural thumb variation alongside distinct Y examples before fitting and fresh-session testing.

## I/Y follow-up adaptation

Four user-labeled I frames were saved under for_I; no Y frames were present. Three I frames were explicitly assigned to adaptation training, one reserved as correlated same-session diagnostic. The reserved I frame ranked I with score 0.8393. Original public validation macro F1 remained 0.9491 and public validation I/Y recalls both remained 1.00. This is not independent webcam accuracy; no new local Y regression test has been completed. Candidate geometry_local_adaptation_I is active locally, with prior weights preserved at models/pre-I-adaptation.joblib. Existing training roles are preserved in data/processed/adaptation-pre-I.json and current roles in data/processed/local-adaptation.json; neither is public. Next: test fresh I with small natural thumb variation and clear Y, without fitting those evaluation frames.
