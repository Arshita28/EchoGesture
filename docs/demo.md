# Local lab demonstration

1. Install requirements and start `python app.py`; open http://127.0.0.1:8080.
2. Confirm model name and supported class count. If untrained, collect/import data and run extraction/training first. Do not demonstrate a fake classifier.
3. Allow webcam access. Use one hand, clearly framed, in training-like lighting. Hold a supported static letter until stable; append manually. Stop releases the webcam.
4. Upload a supported image. Then upload a browser-decodable video showing held static signs. Motion signs and sentence translation are outside scope.
5. Show an empty frame, two hands, and an unfamiliar pose. Explain no-hand/multiple-hand rejection and the limits of score filtering.
6. Show the manifest groups, extraction failures, validation-selected model and untouched test report. Discuss per-letter confusions and small signer sample size.
7. Record live end-to-end responsiveness and recognition errors using the user's actual webcam. Mac hardware speed alone says nothing about accuracy.

If macOS denies camera access, allow the browser in System Settings → Privacy & Security → Camera. Use localhost/127.0.0.1. Uploads must be browser-decodable; convert unsupported HEVC/container combinations to MP4/H.264 if needed. Server uses port 8080 to avoid AirPlay's common port 5000 conflict.

When running in an agent sandbox, MediaPipe on macOS may fail to create an NSOpenGL context. Run locally in Terminal or approve an unsandboxed runtime check. On ordinary local execution no extra network download is needed for the pinned detector.

The primary view is recognition and its model score. Optional manual text assembly is collapsed. Refresh after updates to load the 30% live cutoff slider. The cutoff can be restored to 70% at any time; this leaves all saved evaluation results unchanged. Keep the full right hand visible and the four fingers together for the B reference. Small hand images, changed palm angles and unfamiliar poses can still fail.

Developer diagnosis: after consented examples are manually captured, `python scripts/diagnose.py` compares all three models, handedness metadata, mirrored features and static/tracking behavior using **only the original public training partition for fitting**. Its report stays under ignored `data/processed/`; diagnostic photos are not a new independent test set or automatic training data.
