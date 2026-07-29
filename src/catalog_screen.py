#!/usr/bin/env python3
"""Record TIC 31065777's status in the TESS Ten Thousand Catalog."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from astroquery.vizier import Vizier


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "TIC_31065777_catalog_screen.json"
TIC_ID = "31065777"
CATALOG = "J/ApJS/279/50"
TABLES = {
    "unvetted_unvalidated": f"{CATALOG}/table2",
    "validated_new_eb": f"{CATALOG}/table3",
    "validated_known_eb": f"{CATALOG}/table4",
}


def serialize_row(row) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in row.colnames:
        value = row[name]
        if hasattr(value, "item"):
            value = value.item()
        result[name] = None if getattr(value, "mask", False) else value
    return result


def main() -> int:
    vizier = Vizier(columns=["*"], row_limit=20)
    matches: dict[str, object] = {}
    for label, table in TABLES.items():
        result = vizier.query_constraints(catalog=table, TIC=TIC_ID)
        rows = [serialize_row(row) for returned in result for row in returned]
        matches[label] = {"table": table, "match_count": len(rows), "rows": rows}

    record = {
        "queried_utc": datetime.now(timezone.utc).isoformat(),
        "tic_id": int(TIC_ID),
        "catalog": CATALOG,
        "catalog_article_doi": "10.3847/1538-4365/ade2d8",
        "interpretation": (
            "The target is present in the unvetted and unvalidated supplement, "
            "and absent from the two validated eclipsing-binary tables. This "
            "supports a validation-note framing but is not proof of global novelty."
        ),
        "matches": matches,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
