# Limited ASL phrase recognition — stage 1

## Scope
Separate experimental clip model with ten proposed English output meanings in `config/phrase_vocabulary.json`. These are not verified ASL signing instructions or available training classes. Fluent ASL review and dataset coverage determine the final set. Include NONE for unrelated motion and static gestures. No phrase model is trained yet; existing alphabet weights/routes remain intact.

## Dataset findings (9 November 2026)
- How2Sign: https://how2sign.github.io/ — continuous ASL, English sentence annotations, over 80 hours; CC BY-NC 4.0/research use. Candidate for sentence-data research, but repeated-class coverage for our ten meanings has NOT been established. English transcript matching alone cannot verify semantic equivalence or clip completeness.
- ASL Citizen: https://www.microsoft.com/en-us/research/project/asl-citizen/ — isolated-sign data. Useful for a separate word-recognition baseline; not evidence of sentence translation and not a substitute for complete phrase clips. Review its dataset license before acquisition.

No large video archive downloaded and no fabricated or alphabet-concatenated sentence examples. User is unfamiliar with ASL; fluent review is needed before treating their imitations as ground truth.

## Data contract
Local CSV columns: path,label,signer,session,split,source,usage_basis,asl_reviewed. Keep CSV, clips and derived features under ignored `data/processed/phrases` or `data/raw/phrases`. Path is relative to CSV. Splits: train/validation/test. Keep each signer in one split. Require complete clips with original timing, both hands/upper body/face in view where meaningful, documented permission/license and fluent review. Metadata does not itself prove permission or fluency.

Run `.venv/bin/python scripts/audit_phrase_manifest.py data/processed/phrases/manifest.csv`. The auditor checks required fields, vocabulary, file presence, duplicate content, signer separation and class coverage. Passing this minimum gate does not establish sufficient sample size or quality. Session identity supports later error analysis.

## Implementation order
1. Inspect sentence annotations for actual target coverage, or arrange consented fluent recordings. Finalize supported meanings based on evidence; do not silently replace phrases with isolated words.
2. Extract synchronized left/right hand, upper-body and facial landmarks with missing-point masks and timestamps. Preserve face/body cues; single-hand 63-value alphabet features are insufficient here.
3. Establish a temporal baseline, compare a small sequence network if warranted, select only using validation. Keep weights/features/version separate from alphabet and J/Z.
4. Evaluate signer-held-out clips with per-phrase confusion, uncertainty/rejection rates and complete-pipeline timing. Do not report already-inspected examples as unseen evaluation.
5. Add a separate phrase interface with explicit clip recording boundaries and supported-phrase list only after a model exists. Show unknown rather than forcing every gesture into a phrase. Render the predicted meaning verbatim; do not invent missing words with a text generator.

Next concrete dependency: verified complete-phrase videos with enough repetition and signer diversity. Dataset suitability remains open; there is no phrase-accuracy result yet.
