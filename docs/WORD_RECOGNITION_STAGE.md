# Word recognition stage

## Current milestone

On 16 November 2026, the historical static baseline was reproduced without fitting. Validation macro F1 and test accuracy match the saved report. The active personalized classifier and all 16 local checkpoints remain unchanged. This does not establish new live accuracy. Existing checks and the new WLASL audit checks pass.

Reusable components include Flask routes, camera/media input, captions, speech, dataset provenance patterns and split checks. The current `HandExtractor` intentionally requires exactly one hand. Word recognition needs a separate extractor for two hands and body-relative position; do not relax that rule in the existing alphabet pipeline. Existing J/Z motion features and models are experimental, not a word model.

## Official metadata audit

Source: https://github.com/dxli94/WLASL . Downloaded WLASL_v0.3.json from the official repository on 16 November 2026. Its SHA-256 is recorded in `reports/wlasl-metadata-audit.json`; the downloaded metadata stays local under `datasets/raw/wlasl/`.

| Scope | Listed clips | Train | Validation | Test | Signer IDs across partitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| First 30 official glosses | 709 | 497 | 117 | 95 | 47 |
| WLASL100 | 2038 | 1442 | 338 | 258 | 72 |

These are metadata counts, not downloaded or decoded videos. No clip URLs have been tested. No WLASL videos are present yet. Classes follow official metadata order; no vocabulary was selected using test accuracy or availability. Label IDs are zero based and stable within each declared subset.

Official assignments are preserved, with `val` mapped to `validation`. The measured signer overlap means these subsets cannot support a claim of signer-disjoint evaluation. Keep the official benchmark for comparability and plan a separate fresh-signer camera evaluation; do not silently reshuffle official data to improve scores.

## Repeat the checks

From the EchoGesture root, use a compatible Python environment with the README dependencies. During migration, checks used the original trusted project's `.venv/bin/python`; the new folder has no virtual environment yet.

```sh
python -m pytest -q
python -m scripts.verify_phase1_baseline
python scripts/audit_wlasl.py datasets/raw/wlasl/WLASL_v0.3.json --classes 30
python scripts/audit_wlasl.py datasets/raw/wlasl/WLASL_v0.3.json --classes 100 --output datasets/preprocessed/wlasl100-audit
```

The baseline verifier only loads the trusted preserved model and saved public features; it never trains or overwrites model weights or the historical evaluation report. It refuses mismatched feature hashes. The audit writes local `manifest.csv`, `labels.json` and `audit.json`. Raw URLs, file paths and detailed manifests remain under ignored dataset directories; a small aggregate report is tracked.

For already trimmed local videos, use `--videos PATH` with files named `VIDEO_ID.mp4`. Presence is explicitly marked `present_unverified`. Empty files count as missing. Hash duplicates and missing partitions prevent readiness for the next decode audit. Even complete files never mark training ready automatically.

## Next implementation step

1. Obtain the organizer starter kit to confirm vocabulary, clip type and JSON schema. This can proceed alongside acquisition preparation.
2. Read the official WLASL Computational Use of Data Agreement before using videos. Acquire the first development subset using official download/preprocessing instructions, log unavailable items, and request missing/preprocessed clips if necessary. Do not redistribute source videos in Git.
3. Decode and inspect local clips. Confirm official FPS/frame offsets and whether downloaded videos are raw sources or already trimmed; never trim twice. Record usable duration and frame counts, failures and duplicate hashes.
4. Implement a separate both-hand and upper-body sequence extractor with missing-detection masks. Preserve hand position relative to the body and test its feature consistency.
5. Train a small temporal baseline on the verified development vocabulary, choose settings using validation only, then evaluate the declared held-out subset. Add a no-sign/background strategy before connecting live predictions.

Do not train using metadata alone, report missing videos as usable samples, or describe a 30-class result as WLASL100 performance. Architecture and the website-versus-app decision remain open.

## Acquisition and sequence extraction implementation

The next milestone now has executable tools:

```sh
python scripts/acquire_wlasl.py datasets/raw/wlasl/WLASL_v0.3.json --classes 30 --limit 345 --workers 3
python -m scripts.extract_words datasets/raw/wlasl/WLASL_v0.3.json
python -m scripts.audit_word_sequences
```

`acquire_wlasl.py` uses only official direct HTTPS MP4 URLs that describe full isolated clips (frame start 1 and end -1). It validates every frame and caches verified downloads. It skips YouTube, SWF, HTTP-only and nontrivial trimming entries with explicit reasons. Its caps are 30 MB, 90 seconds and 4,500 decoded frames per clip. It does not disable certificate checks or use mirrors. These restrictions define the initial acquisition scope; skipped clips are not assumed unavailable.

The official WLASL preprocessing copies non-YouTube isolated videos unchanged. Accordingly, acquired full clips are not trimmed again. YouTube sources will need a separately reviewed download and timestamp-based trim path; do not assume their native FPS is 25. The official metadata frame bounds refer to 25 FPS.

`word_features.py` uses the pinned MediaPipe Holistic implementation to associate the signer's left/right hands with the body. It emits 154 float32 features: wrist-relative hand shape, body-relative wrist positions, upper-body image coordinates, visibility values and three presence masks. Shoulder position, size and tilt define a common image-space reference. Hand depth is normalized within each hand; pose and hand depth origins are never subtracted. This is approximate image-space 3D, not calibrated geometry or demonstrated viewpoint invariance. Facial expressions are not represented in this first feature version.

`extract_words.py` samples 32 positions uniformly across each already isolated clip, resets tracking between clips, and saves compressed NPZ files. Missing detections remain zero with explicit masks. No interpolation manufactures missing hands. A provisional quality gate requires body detection on at least 70% and at least one hand on at least 50% of sampled positions. Quality checks establish extraction coverage, not correct signing or recognition accuracy. This desktop Holistic pipeline is a data preparation baseline, not a proven mobile latency or model-size solution.

`audit_word_sequences.py` verifies feature dimensions, hashes, labels, timestamps, presence masks, duplicate content and official partition coverage. It does not automatically remove words or reassign clips to fill missing partitions. Training readiness stays false pending vocabulary and data sufficiency review.

Local acquisition and extraction records remain under ignored `datasets/` paths. Aggregate milestone results are in `reports/word-data-progress.json`. Model checkpoints and the original alphabet application are unchanged. MediaPipe execution may need macOS graphics access outside the sandbox; unit tests use synthetic landmarks and do not require that access.

Sources reviewed: https://github.com/dxli94/WLASL/blob/master/start_kit/preprocess.py ; https://github.com/dxli94/WLASL/blob/master/start_kit/C-UDA-1.0.pdf ; https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/holistic.py . The data-use agreement is retained locally. Source videos are not added to Git.


## Measured acquisition outcome on 16 November 2026

Across all 345 eligible direct URLs, 167 clips passed full decoding. The 178 failures comprise 82 missing files (404), 62 DNS failures and 34 certificate hostname mismatches. The 364 skipped entries require other transport/download/format/trim handling and were not probed by this downloader. No failed host had certificate checks disabled.

All 167 local clips were processed into 32-frame feature sequences; 118 passed provisional detection coverage, with 49 excluded by that gate. Feature integrity checks report no errors. Only **computer, who, deaf, year and cool** currently have quality-passing examples in every official partition. Presence of those partitions alone does not establish sufficient data or accuracy. No word model has been trained. Twenty-seven Python checks pass, including a regression test for cached-feature JSON report serialization.

The next data task is to obtain missing official videos through supported YouTube acquisition/trimming or author/organizer preprocessed clips, then repeat extraction and partition audits. Keep the declared vocabulary fixed; do not hide unavailable classes or move test clips into training. Review detection failures before deciding whether the first feature design needs improvement. Original model checkpoint hashes still match the source project.
