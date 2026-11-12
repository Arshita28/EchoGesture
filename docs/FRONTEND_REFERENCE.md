# Frontend design provenance

Visual reference supplied by the user: https://styles.refero.design/style/e5f5f8cf-e68d-4ed1-bbf5-6b67569af648
Original site: https://dala.craftedbygc.com

The published frontend contains independently written layout/CSS/canvas code, consolidated user-supplied design tokens, and an original procedural particle orb generated mathematically in `static/hero.js`. No reference geometry is needed to run it. No proprietary font is bundled; Helvetica Neue/system fallbacks are used.

An optional local-only `static/brain-particles.json` contains coordinates/colors derived from the original site's public `/images/pos-33.exr` and `/images/cd-33.png` assets. It is excluded from Git and the publication payload because redistribution permission is unconfirmed. A local copy can still supply the existing brain rendering; its absence automatically selects the original procedural art. No reference data is embedded or encoded into the fallback algorithm.

Token formats are consolidated in `static/tokens.css`; invalid verdant hex is corrected to #15846e and range strings are represented as CSS clamp values.
