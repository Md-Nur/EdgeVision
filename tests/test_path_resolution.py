"""
Tests for src/utils/paths.py path-resolution utilities.

Covers:
- get_run_id: v1 (no suffix) vs v2 (_lossv2 suffix)
- resolve_checkpoint_path: correct filename for both versions
- resolve_result_path: correct filename for both versions
- verify_path_exists: raises FileNotFoundError with a message that
  explicitly forbids silent version fallback
- No cross-version fallback: requesting v1 when only v2 exists (or vice versa)
  must raise immediately
"""

import sys
from pathlib import Path

import pytest

# Ensure the project root is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import (
    get_run_id,
    resolve_checkpoint_path,
    resolve_result_path,
    verify_path_exists,
)


# ---------------------------------------------------------------------------
# get_run_id
# ---------------------------------------------------------------------------

class TestGetRunId:
    def test_v1_has_no_suffix(self):
        assert get_run_id("resnet18", seed=42, loss_version="v1") == "resnet18_seed42"

    def test_v2_has_lossv2_suffix(self):
        assert get_run_id("resnet18", seed=42, loss_version="v2") == "resnet18_seed42_lossv2"

    def test_custom_model_v1(self):
        assert get_run_id("custom_none", seed=42, loss_version="v1") == "custom_none_seed42"

    def test_custom_model_v2(self):
        assert get_run_id("custom_none", seed=42, loss_version="v2") == "custom_none_seed42_lossv2"

    def test_different_seed_v2(self):
        assert get_run_id("mobilenetv3_small", seed=7, loss_version="v2") == "mobilenetv3_small_seed7_lossv2"


# ---------------------------------------------------------------------------
# resolve_checkpoint_path
# ---------------------------------------------------------------------------

class TestResolveCheckpointPath:
    def test_v1_checkpoint_filename(self):
        p = resolve_checkpoint_path("resnet18", seed=42, loss_version="v1", checkpoints_dir="checkpoints")
        assert str(p) == "checkpoints/resnet18_seed42_best.pth"

    def test_v2_checkpoint_filename(self):
        p = resolve_checkpoint_path("resnet18", seed=42, loss_version="v2", checkpoints_dir="checkpoints")
        assert str(p) == "checkpoints/resnet18_seed42_lossv2_best.pth"

    def test_v1_returns_path_object(self):
        p = resolve_checkpoint_path("resnet18", seed=42, loss_version="v1", checkpoints_dir="checkpoints")
        assert isinstance(p, Path)

    def test_custom_model_v2(self):
        p = resolve_checkpoint_path("custom_se", seed=42, loss_version="v2", checkpoints_dir="ckpts")
        assert str(p) == "ckpts/custom_se_seed42_lossv2_best.pth"

    def test_custom_model_v1(self):
        p = resolve_checkpoint_path("custom_se", seed=42, loss_version="v1", checkpoints_dir="ckpts")
        assert str(p) == "ckpts/custom_se_seed42_best.pth"

    def test_non_default_seed_v2(self):
        p = resolve_checkpoint_path("resnet18", seed=0, loss_version="v2", checkpoints_dir="checkpoints")
        assert str(p) == "checkpoints/resnet18_seed0_lossv2_best.pth"


# ---------------------------------------------------------------------------
# resolve_result_path
# ---------------------------------------------------------------------------

class TestResolveResultPath:
    def test_v1_result_filename(self):
        p = resolve_result_path("resnet18", seed=42, loss_version="v1", results_dir="results")
        assert str(p) == "results/resnet18_seed42.json"

    def test_v2_result_filename(self):
        p = resolve_result_path("resnet18", seed=42, loss_version="v2", results_dir="results")
        assert str(p) == "results/resnet18_seed42_lossv2.json"

    def test_v1_returns_path_object(self):
        p = resolve_result_path("resnet18", seed=42, loss_version="v1", results_dir="results")
        assert isinstance(p, Path)

    def test_custom_model_v1(self):
        p = resolve_result_path("custom_cbam", seed=42, loss_version="v1", results_dir="res")
        assert str(p) == "res/custom_cbam_seed42.json"

    def test_custom_model_v2(self):
        p = resolve_result_path("custom_cbam", seed=42, loss_version="v2", results_dir="res")
        assert str(p) == "res/custom_cbam_seed42_lossv2.json"


# ---------------------------------------------------------------------------
# verify_path_exists — error behaviour
# ---------------------------------------------------------------------------

class TestVerifyPathExists:
    def test_raises_on_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent_seed42_lossv2_best.pth"
        with pytest.raises(FileNotFoundError):
            verify_path_exists(missing, description="checkpoint", loss_version="v2")

    def test_error_message_mentions_no_fallback(self, tmp_path):
        missing = tmp_path / "ghost_seed42_lossv2_best.pth"
        with pytest.raises(FileNotFoundError, match="(?i)silent fallback"):
            verify_path_exists(missing, description="checkpoint", loss_version="v2")

    def test_error_message_includes_path(self, tmp_path):
        missing = tmp_path / "ghost_seed42_lossv2_best.pth"
        with pytest.raises(FileNotFoundError, match=str(missing)):
            verify_path_exists(missing, description="checkpoint", loss_version="v2")

    def test_no_error_when_file_exists(self, tmp_path):
        existing = tmp_path / "model_seed42_lossv2_best.pth"
        existing.touch()
        # Should not raise
        verify_path_exists(existing, description="checkpoint", loss_version="v2")


# ---------------------------------------------------------------------------
# No cross-version fallback guarantee
# ---------------------------------------------------------------------------

class TestNoVersionFallback:
    def test_requesting_v1_when_only_v2_exists_raises(self, tmp_path):
        """v2 file exists; resolving the v1 path must raise — never silently use v2."""
        (tmp_path / "resnet18_seed42_lossv2_best.pth").touch()
        p = resolve_checkpoint_path("resnet18", seed=42, loss_version="v1", checkpoints_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            verify_path_exists(p, description="checkpoint", loss_version="v1")

    def test_requesting_v2_when_only_v1_exists_raises(self, tmp_path):
        """v1 file exists; resolving the v2 path must raise — never silently use v1."""
        (tmp_path / "resnet18_seed42_best.pth").touch()
        p = resolve_checkpoint_path("resnet18", seed=42, loss_version="v2", checkpoints_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            verify_path_exists(p, description="checkpoint", loss_version="v2")

    def test_both_versions_exist_v1_resolves_correctly(self, tmp_path):
        """When both versions exist, v1 resolution picks v1 file — no contamination."""
        (tmp_path / "resnet18_seed42_best.pth").touch()
        (tmp_path / "resnet18_seed42_lossv2_best.pth").touch()
        p = resolve_checkpoint_path("resnet18", seed=42, loss_version="v1", checkpoints_dir=tmp_path)
        assert p.name == "resnet18_seed42_best.pth"
        verify_path_exists(p, description="checkpoint", loss_version="v1")  # must not raise

    def test_both_versions_exist_v2_resolves_correctly(self, tmp_path):
        """When both versions exist, v2 resolution picks v2 file."""
        (tmp_path / "resnet18_seed42_best.pth").touch()
        (tmp_path / "resnet18_seed42_lossv2_best.pth").touch()
        p = resolve_checkpoint_path("resnet18", seed=42, loss_version="v2", checkpoints_dir=tmp_path)
        assert p.name == "resnet18_seed42_lossv2_best.pth"
        verify_path_exists(p, description="checkpoint", loss_version="v2")  # must not raise

    def test_result_v1_when_only_v2_exists_raises(self, tmp_path):
        """Same guarantee for result JSON files."""
        (tmp_path / "resnet18_seed42_lossv2.json").touch()
        p = resolve_result_path("resnet18", seed=42, loss_version="v1", results_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            verify_path_exists(p, description="result", loss_version="v1")
