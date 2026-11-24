# Broad isolated-word experiment

The production model and this research classifier have separate artifacts. The broad target is the case-insensitive intersection between WLASL glosses and ASL Citizen training labels. A spelling match is a candidate mapping. In particular, different senses of an English word can denote different ASL signs.

## Reproduce the landmark experiment

The source videos and split CSVs come from [ASL Citizen, Desai et al.](https://www.microsoft.com/en-us/research/project/asl-citizen/). The research model uses the [WholeBodyPose release, Lazo-Quispe et al.](https://doi.org/10.1117/12.3096312). Its [135-keypoint release and license](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition/tree/main/ISLR/ASLCITIZEN/ASLCitizen2731) identify CC BY-NC-SA 4.0 and the underlying ASL Citizen terms.

1. Put WLASL metadata in `datasets/raw/wlasl/WLASL_v0.3.json` and the official ASL Citizen split CSVs in `datasets/raw/asl-citizen-selected/`. Existing acquisition scripts use these paths.
2. Download the three ASLCitizen2731 HDF5 files from the release into `datasets/raw/broad-pose/`. Install h5py in the project Python environment.
3. From the project root run `python -m scripts.train_broad_pose`. It verifies the release hashes, checks clip names and glosses against the official partitions, and asserts that participant IDs and filenames are disjoint. Only training clips fit the classifiers. Validation selects the classifier. The test denominator includes unusable and missing clips.
4. The model path is `models/broad-pose-research.joblib`; the metrics path is `reports/broad-pose.json`. The classifier accepts the vectors produced by `describe` in `asl_lab/broad_pose.py`. For a local video, call `describe_video(path)` and pass the vector in a one-element list to the fitted classifier’s `predict` method. Video inference uses MediaPipe joint estimates mapped to the shared anatomy. Differences between RTMPose training landmarks and MediaPipe upload landmarks contribute to cross-source errors.
5. `python -m scripts.evaluate_broad_uploads` scores previously inspected external recordings. For a separate reserved set, first run `python -m scripts.reserve_broad_uploads` with the cached baseline available, then `python -m scripts.evaluate_broad_uploads --reserved`. Selection excludes exact hashes from local clips. It does not establish independence from re-encoded copies or guarantee disjoint external signers.

## Local cache comparison

`python -m scripts.broad_cache_baseline` fits a separate classifier on cached ASL Citizen training clips for WLASL-matching labels. Evaluate with `python -m scripts.evaluate_broad_uploads --cached-baseline`. Its training and video inference both use the project’s MediaPipe preprocessing. The official-attempts denominator includes missing feature caches. Both research experiments keep private upload tests out of training.

## Published RGB comparison

The comparison uses [Microsoft's ASL Citizen I3D model](https://github.com/microsoft/ASL-citizen-code/releases/tag/checkpoints_v1). Put the authors' repository at `datasets/raw/broad-pose/ms-baselines/` and the I3D checkpoint ZIP at `datasets/raw/broad-pose/i3d.zip`. In a Python environment with OpenCV and PyTorch, run `python -m scripts.benchmark_broad_rgb`. The program uses the published test preprocessing with deterministic end padding, maps the 2,731 output indices using Microsoft's sorted training gloss order, and ranks only the 1,159 exact WLASL matches. Results on the fixed external test clips go to `reports/broad-rgb-reserved-uploads.json`. This pretrained model provides a video-level research comparison; its published checkpoint uses floating-point weights.
