# Dataset provenance and collection

## Public candidate: ASLYset v1

- Creator: Miguel Rivera; published 31 October 2019.
- DOI: https://doi.org/10.17632/xs6mvhx6rh.1
- Publisher: https://data.mendeley.com/datasets/xs6mvhx6rh/1
- License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Publisher states 5,200 RGB images, 416×416, four right-handed signers, 50 samples per class/person, 24 static alphabet letters plus SP and FN. Images were captured at roughly 20–40 cm with variation in location.
- Verified metadata: `docs/aslyset-metadata.json`. Archive `ASLYset.rar`, 872,115,989 bytes. SHA-256 `e50202c30dfff4795b88762b765bbeb357e184c13fe22b8f6e0d836d02235c80`.
- Complete archive downloaded and SHA-256 verified. Confirmed 5,200 PNG images, four User folders and 50 images per class/person. Paths include `ASLYset/images/User1/a/User1_A_1.png`; group identity is explicitly encoded. Sampled A/B/C/L/Y images from all four signers were visually inspected. Alphabet subset: 4,800 images, 4,785 successful extractions; see `reports/extraction.json`.

Attribution for redistributed derivatives: “Derived normalized hand landmarks from ASLYset v1, Miguel Rivera (2019), DOI 10.17632/xs6mvhx6rh.1, CC BY 4.0. Changes: excluded SP/FN, extracted MediaPipe coordinates, normalized coordinates, and assigned signer-disjoint partitions.” Retain any third-party notices found in the archive.

```sh
python scripts/download_aslyset.py
tar -xf data/raw/ASLYset.rar -C data/raw
python scripts/import_aslyset.py
python scripts/extract.py data/aslyset-manifest.csv
python -m asl_lab.train
```

The YOLO labels are not model inputs. We use original RGB images for fresh MediaPipe extraction. Exclude SP/FN/J/Z. Candidate partition: User1 and User2 train, User3 validation, User4 test. This is a small four-signer pilot, not evidence of broad population accuracy. Do not randomly split images of the same signer. Inspect hands and label alignment before importing. Extraction failures must be reported by class and signer to expose selection bias.

## Own recordings

Capture with informed consent. Use anonymous IDs; keep identity mapping private. Start with a subset of clearly executed static letters and collect all chosen classes in each partition. Vary distance, lighting, background, camera, and recording day. Keep all sessions from the same signer in the same partition for signer-held-out claims. If evaluating only session generalization, explicitly label that weaker protocol and retain signer IDs separately. Avoid adjacent-frame random splits and augmented duplicates across partitions. This app records individual manually selected frames, not automatic burst videos.

Unknown-hand poses and empty scenes are independent rejection probes, not alphabet labels. Measure their false-accept rate separately. ASLYset only represents a narrow capture setup; webcam data can differ substantially.
