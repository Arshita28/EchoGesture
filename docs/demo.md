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

## Spoken letters

Refresh the page and click **Enable voice**. Only accepted, stable letters are spoken (four accepted frames for video; one accepted prediction for an image). Holding the same letter does not repeat speech. Remove the hand for at least one second, then show the same letter again to repeat it; a different stable letter speaks immediately. A brief uncertain frame does not rearm speech. Disable voice or Stop cancels pending audio. Source/cutoff changes reset speech state. Unsupported browsers show an unavailable message. The selected browser/system voice controls pronunciation and availability; no microphone is requested for speech output.

Implementation reference: https://developer.mozilla.org/en-US/docs/Web/API/Window/speechSynthesis . Functional state tests use a mocked speech engine; audible output still requires a live browser check. Run `node tests/test_voice.cjs` using an available Node runtime to verify repetition and mute behavior.

Voice output now prefers a named female English system voice (Samantha on Apple systems when exposed), or an explicitly female Siri voice if listed by the browser. Other known female voice names are tried next. Browser voice availability varies; Siri is not guaranteed to be exposed. The status displays the selected name or explicitly identifies a system fallback. Voice-selection behavior is covered by a mocked Samantha-versus-Alex check; actual installed voices require a browser listen check.

### Fixed female audio playback

The browser default continued sounding male, so production playback now uses alphabet WAV clips generated locally with the installed macOS Samantha voice. Enable voice previews “Voice enabled” using that same voice. Letter audio uses static/audio/female/; the browser cannot substitute its default voice for these clips. A named-voice speech fallback is allowed only when the HTML Audio API is unavailable; it never chooses an unnamed system default. Script URLs are versioned to replace previously cached voice logic. These clips use Samantha, not Siri's exact voice.

Letter clips were regenerated using explicit spoken letter-name spellings rather than uppercase text, removing macOS's “capital” prefix. Audio and script cache versions were incremented; refresh the page to load the change. The female speech fallback uses the same spoken spellings.
