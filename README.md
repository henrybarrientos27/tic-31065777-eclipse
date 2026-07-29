# TIC 31065777 eclipse candidate

This repository is the reproducible research package for a short Research Note
on five eclipse-like events associated with TIC 31065777 in public TESS
full-frame images.

Public repository: <https://github.com/henrybarrientos27/tic-31065777-eclipse>

Research overview: <https://henrybarrientos27.github.io/tic-31065777-eclipse/>

## Result in one paragraph

Five events in TESS Sectors 8, 9, 35, 36, and 89 follow a linear ephemeris of
`BTJD = 1527.534962 + N * 40.571730224 days`. A 1.5-pixel aperture gives a
median fitted depth of 6.207% and a joint-fit duration of 2.920 hours. The
largest leave-one-sector-out prediction error is 1.90 minutes. Difference-image
missing-light centroids lie 0.16-0.51 TESS pixel from the catalog target
position. These measurements support a recurring eclipse candidate associated
with the target aperture. They do not establish the companion class and do not
fully exclude an unresolved blended eclipsing source.

## Reproduce the analysis

Python 3.12 and the Tectonic LaTeX engine were used for the archived release.
Install Tectonic from https://tectonic-typesetting.github.io/ before running
the complete pipeline. On macOS with Homebrew, use `brew install tectonic`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
```

The first online run downloads public TESSCut pixel files and public MIT
Quick-Look Pipeline light curves from MAST. Later runs can be made without
network access:

```bash
python run_pipeline.py --offline
```

For a faster manuscript-only verification, omit the small detector engineering
benchmark:

```bash
python run_pipeline.py --offline --skip-evaluation
```

The pipeline writes a SHA-256 inventory to
`results/release_manifest.json`. Expected core outputs are:

- `results/TIC_31065777_robust_validation_summary.txt`
- `results/TIC_31065777_centroid_localization.csv`
- `results/TIC_31065777_raw_pixel_lightcurves.csv`
- `results/evidence_packets/TIC_31065777/index.html`
- `results/publication/TIC_31065777_RNAAS_FIGURE.pdf`
- `manuscript/tic_31065777_rnaas.pdf`

## Analysis design

The release performs four distinct checks:

1. It extracts aperture photometry directly from five public TESSCut pixel
   cubes and jointly fits one shared clock and eclipse shape.
2. It refits each event across four aperture radii, omits each sector in turn,
   bootstraps event-center uncertainties, and compares target events with
   spatial and time-shift controls.
3. It measures a missing-light centroid in each sector to test whether the
   dimming is consistent with the TIC position at TESS resolution.
4. It generates an evidence-linked static report and runs a deterministic
   engineering benchmark on synthetic and known-shape fixtures.

The target was selected after examining the data. Consequently, the exact
event-label permutation result in the evidence packet is a post-selection
coherence diagnostic, not a global false-alarm probability. The engineering
benchmark is small and is not an estimate of performance across the stellar
population. Its wide-range blind replication search does not recover the
empirical target, while a preselected narrow-period BLS check does; this is
preserved as an explicit limitation.

## Repository map

```text
data/                 data provenance and optional downloaded pixel files
manuscript/           AASTeX source, bibliography, figure, and compiled note
results/              measurements, controls, evidence packet, and figure
src/                  raw-pixel, robustness, localization, and packet code
tests/                deterministic scientific-contract tests
run_pipeline.py       complete publication pipeline
```

## Data, code, and licenses

The observations are public NASA TESS products served by MAST. Source FITS
files are intentionally excluded from Git because they can be reacquired from
MAST; exact checksums for the analyzed copies are in `data/README.md` and the
release manifest. The derived CSV files contain the data behind the figure.

Code is released under the MIT License. Original prose, figures, and derived
tables in this repository are released under CC BY 4.0. Upstream TESS and QLP
products retain their original terms and attribution requirements.

## Citation

Please cite the archived release described in `CITATION.cff`. After journal
publication, cite the Research Note as the primary scientific reference.

## Contact

Henry Barrientos - `henrywaynebarrientos@gmail.com`
