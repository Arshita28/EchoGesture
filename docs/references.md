# Reference implementations and reuse review

Reviewed primary repositories on 7–8 November 2026:

1. https://github.com/kinivi/hand-gesture-recognition-mediapipe — Apache-2.0. Inspected README and `app.py`. Relevant ideas: wrist-relative normalized coordinates, collecting labeled keypoints, and temporal gesture history. Its bundled labels are generic open/closed/point gestures, not an ASL alphabet dataset; its MLP/TFLite artifacts and older dependency stack are not reused. This project independently implements 3D max-radius normalization, explicit signer partitions and classical classifiers.
2. https://github.com/ecabestadistica/ASL-App — code LICENSE is **CC BY-NC 4.0**, distinct from the linked dataset's CC BY 4.0. Inspected `app_hybrid.py`, README and LICENSE. Useful comparison: 24 static ASL classes, webcam/video paths, stage timing and temporal smoothing. DenseNet image/skeleton models and no-hand fallback differ from this syllabus-oriented pipeline; no source, models or example images copied. Reported repository speed/accuracy does not establish this project's performance.
3. https://github.com/google-ai-edge/mediapipe — pretrained hand detector dependency. Documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python . Current documentation emphasizes Tasks; this project pins 0.10.21 for the bundled legacy Hands API verified on this machine. Upgrade deliberately and re-extract/retest if preprocessing changes.
4. https://scikit-learn.org/stable/ — classical estimators, pipelines and evaluation metrics. Group identities are explicitly assigned before extraction here rather than trusting frame-wise random splitting.

Architecture screenshot/reference course: https://www.krishnaik.in/project/sign-language-detection-systems-for-deaf-and-mute-individuals . Its YOLOv5/Flask/AWS approach is background reference; YOLO, cloud hosting and CI/CD are not required for the local landmark lab.

No external project source code is vendored. Package licenses remain with their distributions. Public dataset attribution is documented separately in `dataset.md`. GitHub account provided by user: https://github.com/MTJ19 ; no publication or remote ownership is inferred from the profile link alone.
