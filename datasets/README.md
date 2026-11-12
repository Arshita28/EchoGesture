# Datasets used by Sign Studio

This is the canonical dataset folder. Start with [DATASET_REPORT.md](DATASET_REPORT.md) for sources, exact counts, preprocessing, exclusions and training roles. Counts were audited from disk on **9 November 2026**; they are not publisher claims or model-accuracy estimates.

```text
datasets/
├── raw/                         Original downloaded/collected media
│   ├── ASLYset.rar              Complete original public archive
│   ├── ASLYset/                 Extracted original images and supplied annotations
│   ├── signalpha/               Downloaded public motion-video subset
│   ├── collected/               Local manually captured webcam images
│   └── uploads/                 Local upload/video debugging media
├── preprocessed/                Extracted landmarks and development artifacts
│   ├── landmarks.jsonl          4,785 public static feature rows
│   ├── landmarks.extraction.json
│   ├── motion-public/           76 public landmark sequences and extraction report
│   ├── motion/                  26 local landmark sequences
│   ├── local-adaptation.json    Actual selected local static training/diagnostic roles
│   ├── motion-balanced-adaptation.json
│   └── *.json                   Historical diagnostics and experiment reports
├── aslyset-manifest.csv         4,800 selected raw static-image references
├── manifest.csv                 71 local image references with original capture labels
├── documentation/              Publisher metadata and original baseline report snapshots
├── inventory/                  Local audit files; includes private filenames
│   ├── files.csv               Relative path, bytes and SHA-256 for each physical file
│   └── summary.json            Audited counts and role summary
├── DATASET_REPORT.md
└── processed -> preprocessed   Compatibility alias; not another copy
```

The project-root `data -> datasets` link preserves existing scripts, model records and Flask capture paths. Use `datasets/raw` and `datasets/preprocessed` when browsing. Aliases do not duplicate the data; avoid following both when counting files. Collection continues to save into this canonical folder through the existing alias.

The inventory covers 10,701 physical files and 1,877,738,002 bytes (about 1.75 GiB), including the retained archive, original annotations and diagnostic JSON. This total is **not a sample count**. The manifest CSVs and documentation are outside that media/processed inventory.

Raw media, landmark data, private manifests, detailed inventories and trained models remain excluded from Git. Documentation contains aggregate counts and public provenance. No original data was removed or retrained during organization. `models/` remains separate because models are learned artifacts, not preprocessed datasets.

Verify all inventory hashes after a move:

```sh
.venv/bin/python scripts/verify_dataset_inventory.py
```

New captures after this snapshot will be reported as additional files; the snapshot is not automatically rewritten. Historical filenames and stored partition labels are preserved for traceability. Consult the adaptation-role records before making training/test claims.
