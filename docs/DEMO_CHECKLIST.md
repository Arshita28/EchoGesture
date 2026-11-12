# Five-minute demonstration

## Before starting
- Run `python app.py`; open http://127.0.0.1:8080/sign.
- Check camera access, lighting and full-hand framing. Test audio once.
- Have the original benchmark table available. Label it as the original public-dataset benchmark.

## Demonstration sequence
1. **Purpose (30 seconds):** Explain that MediaPipe provides landmarks and the trained classical model classifies isolated ASL letters.
2. **Recognition (60 seconds):** Show three static letters that work in the current setup. Also remove the hand to demonstrate clearing. If uncertain, report the result honestly.
3. **Message (60 seconds):** Compose `HELLO` using manual append, including two L clicks. Add a space, demonstrate deletion/correction, and speak the message. If the browser lacks a local female voice, show the availability message and copy the text.
4. **ML evidence (60 seconds):** Compare SVM/RF/KNN validation selection and the original test table. Explain 90.98% original SVM accuracy versus 69.37% coverage at 70% filtering. Neither describes current live accuracy.
5. **Evaluation (45 seconds):** Open `/sign?tools=1`, run one three-second trial, and show its readable report. One trial is not overall accuracy.
6. **Limits and next steps (45 seconds):** Static 24-letter scope, user-dependent behavior, manual message assembly, unresolved J/Z. Explain that fresh-user evaluation is next.

## Questions to prepare for
- Why landmarks instead of raw pixels? Compact features make classical models practical; MediaPipe still contains pretrained deep learning.
- Why split by signer? Random adjacent-frame splits leak highly similar hand appearances across partitions.
- Why SVM? It had the strongest validation macro F1 in the original comparison; this does not prove it is universally best.
- Why not claim 99.64% accuracy? That number covers accepted original-test predictions only, with many attempts rejected.
- Why does J/Z need another method? A single pose cannot encode the necessary movement trajectory.
- Is this sentence translation? No. The user manually composes letters and word boundaries.
