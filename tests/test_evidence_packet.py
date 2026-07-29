from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import sqlite3

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evidence_packet.analysis import (
    bootstrap_ephemeris,
    harmonic_period_agreement,
    robust_replication_ephemeris,
    weighted_ephemeris,
)
from evidence_packet.generator import generate_evidence_packet
from evidence_packet.evaluation import _synthetic_frame


class ScientificAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.cycles = np.asarray([0, 1, 18, 19, 54], dtype=float)
        self.epoch = 1527.535
        self.period = 40.57173
        self.centers = self.epoch + self.cycles * self.period
        self.errors = np.full(len(self.cycles), 0.001)

    def test_weighted_ephemeris_recovers_known_period(self):
        result = weighted_ephemeris(self.cycles, self.centers, self.errors)
        self.assertAlmostEqual(result["period_days"], self.period, places=10)
        self.assertAlmostEqual(result["epoch_btjd"], self.epoch, places=10)

    def test_replication_is_separate_and_agrees(self):
        result = robust_replication_ephemeris(self.cycles, self.centers)
        self.assertEqual(
            result["implementation"], "median_pairwise_slopes_and_median_intercept"
        )
        self.assertAlmostEqual(result["period_days"], self.period, places=10)
        self.assertTrue(harmonic_period_agreement(self.period, result["period_days"]))

    def test_bootstrap_is_reproducible(self):
        first, first_summary = bootstrap_ephemeris(
            self.cycles,
            self.centers,
            self.errors,
            trials=30,
            seed=123,
        )
        second, second_summary = bootstrap_ephemeris(
            self.cycles,
            self.centers,
            self.errors,
            trials=30,
            seed=123,
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first_summary, second_summary)

    def test_corrupted_fixture_has_point_level_truth_ledger(self):
        frame, corruption = _synthetic_frame(
            model="eclipse",
            seed=987,
            depth=0.02,
            corrupted=True,
        )
        self.assertGreater(len(frame), 0)
        self.assertGreater(len(corruption), 0)
        self.assertEqual(
            set(corruption["corruption"]),
            {"nonfinite_flux", "nonfinite_time", "additive_flux_outlier"},
        )
        removed = ~corruption["retained_after_finite_filter"].astype(bool)
        self.assertTrue(removed.any())

class PacketContractTest(unittest.TestCase):
    def test_cached_tic_31065777_packet_contract(self):
        cached = list(
            (PROJECT_ROOT / "results" / "omega_v2").glob(
                "*/tables/TIC_31065777_lightcurve.parquet"
            )
        )
        if not cached:
            self.skipTest("TIC 31065777 public-data cache is not present")
        with tempfile.TemporaryDirectory() as temporary:
            output = generate_evidence_packet(
                31065777,
                project_root=PROJECT_ROOT,
                output_dir=Path(temporary) / "packet",
                bootstrap_trials=30,
                seed=123,
                offline=True,
            )
            required = [
                "index.html",
                "evidence.sqlite",
                "artifact_manifest.csv",
                "data/provenance.json",
                "data/quality_control.csv",
                "data/cleaning_decisions.csv",
                "data/period_search.csv",
                "data/signal_measurements.json",
                "data/instrument_comparisons.csv",
                "data/significance_tests.json",
                "data/injection_tests.csv",
                "data/held_out_validation.csv",
                "data/replication.json",
                "data/replication_period_search.csv",
                "data/claim_evidence.csv",
                "data/conventional_explanations.json",
                "data/unresolved_weaknesses.json",
                "data/claims.json",
                "plots/phase_fold.png",
            ]
            for relative in required:
                self.assertTrue((output / relative).is_file(), relative)
            dispositions = pd.read_csv(output / "data/cadence_disposition.csv")
            qc = pd.read_csv(output / "data/quality_control.csv")
            self.assertTrue(
                {
                    "source_flux",
                    "source_flux_column",
                    "decision_rule_id",
                    "reason",
                }.issubset(dispositions.columns)
            )
            self.assertEqual(len(dispositions), int(qc["raw_points"].sum()))
            self.assertEqual(
                int((dispositions["disposition"] == "removed").sum()),
                int(qc["removed_points"].sum()),
            )
            self.assertEqual(int(qc["cache_mismatch_points"].sum()), 0)
            claim_evidence = pd.read_csv(output / "data/claim_evidence.csv")
            self.assertTrue(claim_evidence["artifact_exists"].astype(bool).all())
            self.assertTrue(claim_evidence["selector_valid"].astype(bool).all())
            numerical = claim_evidence[
                claim_evidence["claim_class"].isin(["measurement", "interpretation"])
            ]
            self.assertTrue(
                (
                    numerical.groupby("claim_id")["numeric_value_count"].sum() > 0
                ).all()
            )
            with sqlite3.connect(output / "evidence.sqlite") as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            self.assertTrue(
                {
                    "claims",
                    "signal_measurements",
                    "significance_tests",
                    "replication",
                    "provenance",
                    "reproducibility",
                    "claim_evidence",
                    "replication_period_search",
                }.issubset(tables)
            )


if __name__ == "__main__":
    unittest.main()
