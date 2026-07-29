# Literature and catalog screen

Checked 2026-07-28. This is a reproducible scope check, not a claim that every
possible publication or unpublished observing program has been searched.

## Decisive catalog result

An exact TIC query of the TESS Ten Thousand Catalog (Kostov et al. 2025,
ApJS 279, 50; DOI `10.3847/1538-4365/ade2d8`) returned:

- one TIC 31065777 row in VizieR table `J/ApJS/279/50/table2`, the list of
  unvetted and unvalidated neural-network targets;
- zero rows in `table3`, the 7,936 validated new eclipsing binaries; and
- zero rows in `table4`, the 2,065 validated previously known eclipsing
  binaries.

The machine-readable query result is
`results/TIC_31065777_catalog_screen.json`, and the query can be refreshed with
`python src/catalog_screen.py`.

Primary sources:

- Article: https://doi.org/10.3847/1538-4365/ade2d8
- VizieR catalog: https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/279/50

## Publication framing

The defensible contribution is not an unrestricted first-discovery claim. It is
the pixel-level validation, five-event ephemeris, robustness analysis, and
Sector 89 extension of a target that was previously distributed only as an
unvetted machine-learning candidate. Catalog absence elsewhere cannot prove
novelty, and the manuscript does not make that claim.
