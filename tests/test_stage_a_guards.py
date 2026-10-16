"""
Tests for Stage A Dry-Run Isolation, Aggregator Safeguards, and Collapse Guard.
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from src.orchestrator import check_collapse_guard, is_run_completed
from src.aggregate_results import aggregate_stage_a


CLASS_NAMES = [
    "Papaya___Anthracnose",
    "Papaya___Healthy",
    "Papaya___Leaf_Curl",
    "Papaya___Ringspot",
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Potato___Leafroll_Virus",
    "Rice___Bacterial_Leaf_Blight",
    "Rice___Brown_Spot",
    "Rice___Healthy",
    "Rice___Tungro",
]


class TestCollapseGuard:
    """Test suite for collapse guard under various failure modes."""

    def test_guard_inactive_before_min_epochs(self):
        """Collapse guard should not trigger before 10 epochs even if F1 is low."""
        val_res = {
            "macro_f1": 0.05,
            "y_pred": [0] * 100,
        }
        is_collapsed, reason, hist = check_collapse_guard(
            val_res, epoch=5, class_names=CLASS_NAMES, min_epochs=10
        )
        assert not is_collapsed
        assert reason == ""
        assert hist == {}

    def test_guard_healthy_model(self):
        """Collapse guard should not trigger for healthy predictions."""
        # Balanced predictions across classes, high F1
        val_preds = list(range(12)) * 10
        val_res = {
            "macro_f1": 0.85,
            "y_pred": val_preds,
        }
        is_collapsed, reason, hist = check_collapse_guard(
            val_res, epoch=11, class_names=CLASS_NAMES, min_epochs=10
        )
        assert not is_collapsed
        assert reason == ""
        assert len(hist) == 12

    def test_guard_triggers_on_low_macro_f1(self, capsys):
        """Collapse guard triggers if validation macro-F1 < 0.2 after 10 epochs."""
        # Distributed predictions but poor accuracy/F1
        val_preds = [0, 1, 2, 3] * 25
        val_res = {
            "macro_f1": 0.12,
            "y_pred": val_preds,
        }
        is_collapsed, reason, hist = check_collapse_guard(
            val_res, epoch=10, class_names=CLASS_NAMES, min_epochs=10
        )
        assert is_collapsed
        assert "Validation macro-F1 < 0.2" in reason
        assert hist["Papaya___Anthracnose"] == 25

        captured = capsys.readouterr()
        assert "COLLAPSE GUARD TRIGGERED at Epoch 10" in captured.out
        assert "Validation Prediction Histogram:" in captured.out

    def test_guard_triggers_on_single_class_dominance(self, capsys):
        """Collapse guard triggers if single class is predicted for >90% of val images."""
        # 95 out of 100 predictions are class 4 (Potato___Early_Blight)
        val_preds = [4] * 95 + [0, 1, 2, 3, 5]
        val_res = {
            "macro_f1": 0.25,  # even if F1 >= 0.2
            "y_pred": val_preds,
        }
        is_collapsed, reason, hist = check_collapse_guard(
            val_res, epoch=12, class_names=CLASS_NAMES, min_epochs=10
        )
        assert is_collapsed
        assert "Dominant single class" in reason
        assert "Potato___Early_Blight" in reason
        assert hist["Potato___Early_Blight"] == 95

        captured = capsys.readouterr()
        assert "COLLAPSE GUARD TRIGGERED at Epoch 12" in captured.out
        assert "Potato___Early_Blight" in captured.out


class TestAggregatorSafeguards:
    """Test suite for aggregator refusing dry directories and warning on <10 epochs."""

    def test_aggregator_refuses_dry_results_dir(self):
        """Aggregator must raise ValueError when directed to results_dry."""
        with pytest.raises(ValueError, match="REFUSED"):
            aggregate_stage_a(results_dir="results_dry")

    def test_aggregator_refuses_dry_logs_dir(self):
        """Aggregator must raise ValueError when logs_dir points to dry directories."""
        with pytest.raises(ValueError, match="REFUSED"):
            aggregate_stage_a(results_dir="results", logs_dir="logs_dry")

    def test_aggregator_warns_on_under_10_epochs(self, tmp_path, capsys):
        """Aggregator must print warning if any run has fewer than 10 epochs logged."""
        res_dir = tmp_path / "custom_results"
        logs_dir = tmp_path / "custom_logs"
        res_dir.mkdir()
        logs_dir.mkdir()

        # Create a mock run with 4 epochs
        run_data = {
            "model": "resnet18",
            "seed": 42,
            "dry_run": False,
            "epochs_logged": 4,
            "standard_test": {"macro_f1": 0.90, "accuracy": 0.91, "y_true": [0, 1], "y_pred": [0, 1], "group_ids": [1, 2]},
            "dedup_test": {"macro_f1": 0.89, "accuracy": 0.90, "y_true": [0, 1], "y_pred": [0, 1], "group_ids": [1, 2]},
            "parameter_count": 1000,
            "model_size_mb": 4.0,
            "gflops": 1.0,
            "cpu_latency_ms": 5.0,
        }
        with open(res_dir / "resnet18_seed42.json", "w") as f:
            json.dump(run_data, f)

        # Call aggregator on this directory
        aggregate_stage_a(
            results_dir=res_dir,
            output_table_path=res_dir / "table.csv",
            reports_dir=tmp_path / "reports",
            logs_dir=logs_dir,
            n_bootstraps=10,
        )

        captured = capsys.readouterr()
        assert "fewer than 10 epochs logged (4 epochs)" in captured.out


class TestResumabilityIsolation:
    """Test suite ensuring dry runs are isolated from production results."""

    def test_is_run_completed_rejects_dry_run_in_prod(self, tmp_path):
        """is_run_completed with dry_run=False must reject dry run JSONs."""
        dry_json = tmp_path / "results_dry" / "resnet18_seed42.json"
        dry_json.parent.mkdir(parents=True)
        with open(dry_json, "w") as f:
            json.dump({
                "dry_run": True,
                "standard_test": {"macro_f1": 0.8},
                "dedup_test": {"macro_f1": 0.8},
                "cpu_latency_ms": 10.0,
                "gflops": 1.0,
            }, f)

        # In production mode (dry_run=False), must return False
        assert not is_run_completed(dry_json, dry_run=False)

    def test_is_run_completed_accepts_valid_prod_run(self, tmp_path):
        """is_run_completed with dry_run=False accepts full verified runs."""
        prod_json = tmp_path / "results" / "resnet18_seed42.json"
        prod_json.parent.mkdir(parents=True)
        with open(prod_json, "w") as f:
            json.dump({
                "dry_run": False,
                "standard_test": {"macro_f1": 0.95},
                "dedup_test": {"macro_f1": 0.95},
                "cpu_latency_ms": 10.0,
                "gflops": 1.0,
            }, f)

        assert is_run_completed(prod_json, dry_run=False)
