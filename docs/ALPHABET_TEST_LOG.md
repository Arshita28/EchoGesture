# User webcam alphabet test — 8 November 2026

Qualitative user observations at the current live cutoff, not a controlled accuracy measurement. Preserve as a regression checklist. J/Z remain deferred.

| Letter | Observed behavior |
|---|---|
| A | Working |
| B | Working; sometimes low score |
| C | Correct candidate; consistently uncertain |
| D | Orientation-dependent; otherwise predicts X; captures saved |
| E | Predicts S |
| F | Working with low score |
| G | Usually uncertain with very low score |
| H | Working |
| I | Working |
| K | Working |
| L | Working |
| M | Working after local adaptation |
| N | Working |
| O | Working |
| P | Confuses with H; captures saved |
| Q | Not working; wrong prediction unspecified |
| R | Working |
| S | Working |
| T | Working |
| U | Confuses with K/V |
| V | Mostly working; sometimes predicts K |
| W | Working |
| X | Confuses with P/G |
| Y | Working |

Priorities: reproduce D/P using existing local captures; collect E/Q/U/X/G and uncertain C failures; distinguish top-label errors from low-score abstention and detector loss. Recheck all reported working letters after each candidate. No hand-coded label substitutions or blanket cutoff reduction. New training captures must be assigned explicit roles; a fresh later recording is required for evaluation, since adjacent frames are correlated and all captures are from the same user.
