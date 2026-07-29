# Data provenance

This analysis uses only public observations served by the Mikulski Archive for
Space Telescopes (MAST).

## Raw-pixel products

`run_pipeline.py` requests 15 by 15 pixel TESSCut target-pixel files at
RA = 135.070041926 degrees and Dec = -45.052218111 degrees for TESS Sectors 8,
9, 35, 36, and 89. The analyzed local copies had these SHA-256 hashes:

```text
Sector 8   adae9caf639cea0950c2363360b40327d33ed8b92b35ba316888ef649eefe2f1
Sector 9   c0ab10d70c37fd6e93f4f0597550ca2e9c3137b5afc0ceb5c0e4f1fc92dbc369
Sector 35  2de89d6f15e00aec63998166b3013b17ac37d26b078deb65c4edbe18a22e1d69
Sector 36  37dfea68cde818179810c1e97f6cc0d159e89ce88f4c1c7f2e76dd20b7a7d13d
Sector 89  6fc7e7b294999edf12be23e8936b3b37e94e34ac82386403d0aff529319674d2
```

Archive service: https://mast.stsci.edu/tesscut/

## Survey light curves

The evidence packet and phase-folded context plot use public MIT Quick-Look
Pipeline TESS full-frame-image light curves distributed as a MAST High-Level
Science Product. Exact source-product names, flux columns, byte sizes, and
SHA-256 hashes are recorded in
`results/evidence_packets/TIC_31065777/data/provenance.json`.

QLP archive DOI: https://doi.org/10.17909/t9-r086-e880

## Time and flux conventions

- Times are TESS Barycentric Julian Date (BTJD = BJD - 2457000).
- Raw-pixel aperture fluxes are background-subtracted and normalized to the
  local out-of-event median.
- The publication depth is the median of five fitted depths from the 1.5-pixel
  aperture robustness run.
- Derived, figure-level data are distributed in `results/*.csv`.
