# Public motion training data candidates — 8 November 2026

## Preferred candidate: SignAlphaSet v3

Publisher: https://data.mendeley.com/datasets/8fmvr9m98w/3
DOI: 10.17632/8fmvr9m98w.3
Creators: Bindu Garg, Manisha Kasar, Achyut Kashyap, Amber Vats, Gunjan Sharma, Aditya Hange.
License verified in publisher page and API: CC BY 4.0. Credit creators, DOI/license and preprocessing changes when reusing derivatives. Metadata snapshot: docs/signalphaset-metadata.json.

API confirms ASL_dynamic.zip is available, size 1,247,063,491 bytes (about 1.25 GB). A separate static image archive exists; static frames alone cannot establish J/Z motion recognition. The publisher describes alphabet A–Z video folders and five common gestures. Its headline says 300 videos while another statement describes 31 folders with ten videos each; actual counts must be audited rather than assumed.

Not downloaded/extracted/trained yet. Before importing: inspect J/Z clips to verify real movement and correct ASL labels; inspect signer/session identity and any duplicate frames/recordings; confirm MediaPipe extraction and temporal boundaries; preserve complete videos and signer groups across partitions. The metadata summary alone does not establish signer-disjoint training is possible. Existing static letters/unrelated videos may supply background candidates, but background behavior must be checked and balanced.

Potentially useful to bootstrap training so the user need not record the full training corpus. Still requires independent local webcam tests. Do not claim downloaded/trained model accuracy from the publisher's reported results.

## Larger alternative: ChicagoFSWild

Primary source: https://home.ttic.edu/~klivescu/ChicagoFSWild.htm
Naturally occurring ASL fingerspelling sequences with signer/partition metadata. Source advertises a 14 GB archive and larger extension. Sequence labels correspond to whole fingerspelled strings, so isolated J/Z requires additional temporal alignment/segmentation. Review actual use terms before acquisition. Less direct for the current whole-clip J/Z classifier.

Do not substitute Indonesian, Indian, Mexican or other sign-language alphabet datasets merely because letter names match ASL. ASL MNIST excludes motion J/Z and is not a source of motion training clips.
