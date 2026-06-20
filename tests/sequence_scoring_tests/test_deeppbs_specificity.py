"""tests/sequence_scoring_tests/test_deeppbs_specificity.py.

Tests for DeepPBS specificity tool wrapper, canonicalization, and fallback.
"""

from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from proto_tools.tools.sequence_scoring.deeppbs_specificity import (
    DeepPBSSpecificityConfig,
    DeepPBSSpecificityInput,
    run_deeppbs_specificity,
)
from proto_tools.tools.tool_registry import ToolRegistry
from tests.tool_infra_tests.test_export_functionality import validate_output

# ── Registration ──────────────────────────────────────────────────────────────


def test_tool_key_registered():
    """Tool key is visible in ToolRegistry under sequence_scoring."""
    spec = ToolRegistry.get("deeppbs-specificity")
    assert spec.key == "deeppbs-specificity"
    assert spec.category == "sequence_scoring"


# ── Validation ────────────────────────────────────────────────────────────────


def test_input_accepts_single_path_string():
    """Single string path is normalized to a list."""
    payload = DeepPBSSpecificityInput(pdb_paths="one.pdb")
    assert payload.pdb_paths == ["one.pdb"]


def test_input_rejects_empty_list():
    """Empty pdb_paths is rejected."""
    with pytest.raises(ValidationError, match="pdb_paths must contain at least one path"):
        DeepPBSSpecificityInput(pdb_paths=[])


def test_input_rejects_extra_fields():
    """Unknown input fields are rejected."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DeepPBSSpecificityInput(pdb_paths=["one.pdb"], extra_field="x")


def test_config_defaults_resolve_local_paths():
    """Config exposes machine-local DeepPBS + X3DNA defaults."""
    cfg = DeepPBSSpecificityConfig()
    assert cfg.deeppbs_repo_path.endswith("DeepPBS")
    assert cfg.x3dna_bin_path is not None and cfg.x3dna_bin_path.endswith("DSSR")


def test_config_cloud_unsupported():
    """DeepPBS cannot run on device='cloud' (needs local repo/X3DNA)."""
    assert DeepPBSSpecificityConfig().cloud_unsupported_reason() is not None


# ── Dispatch (mocked) ─────────────────────────────────────────────────────────


@patch("proto_tools.tools.sequence_scoring.deeppbs_specificity.deeppbs_specificity.ToolInstance.dispatch")
def test_run_calls_dispatch_and_returns_output(mock_dispatch):
    """run_deeppbs_specificity returns canonicalized results."""
    mock_dispatch.return_value = {
        "results": [
            {
                "input_name": "candidate_0",
                "source_method": "deeppbs",
                "output_npz_path": "candidate_0.npz",
                "predicted_ppm": [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]],
                "true_sequence": [0, 1],
                "mask": [1, 1],
                "dna_mask": [1, 1],
                "chain_labels": [0, 1],
                "used_fallback": False,
                "fallback_reason": None,
            }
        ]
    }

    output = run_deeppbs_specificity(
        DeepPBSSpecificityInput(pdb_paths=["candidate_0.pdb"]),
        DeepPBSSpecificityConfig(),
    )
    validate_output(output)

    assert len(output.results) == 1
    assert output.results[0].source_method == "deeppbs"
    assert output.results[0].true_sequence == [0, 1]
    assert output.results[0].used_fallback is False
    mock_dispatch.assert_called_once()


@patch("proto_tools.tools.sequence_scoring.deeppbs_specificity.deeppbs_specificity.ToolInstance.dispatch")
def test_run_surfaces_fallback_fields(mock_dispatch):
    """Fallback results round-trip used_fallback / fallback_reason through the wrapper."""
    mock_dispatch.return_value = {
        "results": [
            {
                "input_name": "candidate_0",
                "source_method": "deeppbs",
                "output_npz_path": "candidate_0.npz",
                "predicted_ppm": [[0.25, 0.25, 0.25, 0.25]],
                "true_sequence": [0],
                "mask": [1],
                "dna_mask": [1],
                "chain_labels": [0],
                "used_fallback": True,
                "fallback_reason": "DeepPBS dependency missing: x3dna-dssr not found in PATH.",
            }
        ]
    }

    output = run_deeppbs_specificity(
        DeepPBSSpecificityInput(pdb_paths=["candidate_0.pdb"]),
        DeepPBSSpecificityConfig(),
    )
    validate_output(output)
    assert output.results[0].used_fallback is True
    assert "x3dna-dssr" in output.results[0].fallback_reason


@patch("proto_tools.tools.sequence_scoring.deeppbs_specificity.deeppbs_specificity.ToolInstance.dispatch")
def test_dispatch_errors_propagate(mock_dispatch):
    """Dispatch failures propagate (decorator owns error policy; raises by default)."""
    mock_dispatch.side_effect = RuntimeError("DeepPBS failed")
    with pytest.raises(RuntimeError, match="DeepPBS failed"):
        run_deeppbs_specificity(
            DeepPBSSpecificityInput(pdb_paths=["candidate_0.pdb"]),
            DeepPBSSpecificityConfig(),
        )


# ── Canonicalization (pure standalone logic) ──────────────────────────────────


def test_canonicalize_prediction_transforms_to_schema(tmp_path):
    """Raw DeepPBS arrays are converted to canonical DNA schema."""
    from proto_tools.tools.sequence_scoring.deeppbs_specificity.standalone.run import (
        _canonicalize_prediction,
    )

    raw_path = tmp_path / "raw_predict.npz"
    out_path = tmp_path / "canonical.npz"

    pred = np.array([[0.70, 0.10, 0.10, 0.10], [0.10, 0.70, 0.10, 0.10]], dtype=np.float64)
    seq = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    np.savez_compressed(raw_path, P=pred, Seq=seq)

    payload = _canonicalize_prediction(str(raw_path), str(out_path))

    # Concatenates forward + mirrored strand, so length doubles.
    assert len(payload["predicted_ppm"]) == 4
    assert len(payload["predicted_ppm"][0]) == 4
    assert len(payload["true_sequence"]) == 4
    assert payload["mask"] == [1, 1, 1, 1]
    assert payload["dna_mask"] == [1, 1, 1, 1]
    assert out_path.exists()
    # Each row of the PPM is normalized to sum to 1.
    for row in payload["predicted_ppm"]:
        assert abs(sum(row) - 1.0) < 1e-9


# ── Fallback (pure standalone logic) ──────────────────────────────────────────


def test_fallback_canonical_from_pdb_builds_uniform_ppm(tmp_path):
    """Fallback derives a uniform PPM and DNA truth from PDB DNA residues."""
    from proto_tools.tools.sequence_scoring.deeppbs_specificity.standalone.run import (
        _fallback_canonical_from_pdb,
    )

    pdb_path = tmp_path / "complex.pdb"
    # Two DNA residues (DA, DC) on chain B; a protein residue (ALA) is ignored.
    pdb_path.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  P    DA B   1       1.000   1.000   1.000  1.00  0.00           P\n"
        "ATOM      3  P    DC B   2       2.000   2.000   2.000  1.00  0.00           P\n"
    )
    out_path = tmp_path / "fallback.npz"

    payload = _fallback_canonical_from_pdb(str(out_path), str(pdb_path))

    assert payload["true_sequence"] == [0, 1]  # DA -> 0, DC -> 1
    assert payload["chain_labels"] == [0, 0]
    assert payload["mask"] == [1, 1]
    assert payload["dna_mask"] == [1, 1]
    assert payload["predicted_ppm"] == [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]
    assert out_path.exists()


def test_fallback_canonical_handles_no_dna_residues(tmp_path):
    """Fallback degrades gracefully when no DNA residues are present."""
    from proto_tools.tools.sequence_scoring.deeppbs_specificity.standalone.run import (
        _fallback_canonical_from_pdb,
    )

    pdb_path = tmp_path / "protein_only.pdb"
    pdb_path.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
    out_path = tmp_path / "fallback.npz"

    payload = _fallback_canonical_from_pdb(str(out_path), str(pdb_path))

    assert payload["true_sequence"] == [0]
    assert payload["predicted_ppm"] == [[0.25, 0.25, 0.25, 0.25]]


# ── Integration (real DeepPBS env; skipped by default) ────────────────────────


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_deeppbs_specificity_runs_on_structure():
    """Run DeepPBS end-to-end on a real protein-DNA PDB (requires local env + X3DNA)."""
    import os

    pdb = os.environ.get("DEEPPBS_TEST_PDB")
    if not pdb or not os.path.exists(pdb):
        pytest.skip("Set DEEPPBS_TEST_PDB to a protein-DNA PDB to run this integration test")

    output = run_deeppbs_specificity(
        DeepPBSSpecificityInput(pdb_paths=[pdb]),
        DeepPBSSpecificityConfig(),
    )
    validate_output(output)
    assert len(output.results) == 1
    result = output.results[0]
    # PPM is L x 4 in ACGT order; rows normalized.
    assert all(len(row) == 4 for row in result.predicted_ppm)
    assert os.path.exists(result.output_npz_path)


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("deeppbs-specificity")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_deeppbs_specificity_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark deeppbs-specificity on a real protein-DNA structure (cold + warm).

    Requires a local DeepPBS repository, an X3DNA install, and the standalone
    env. Skips cleanly when ``DEEPPBS_TEST_PDB`` is unset or the env is absent.
    """
    import os

    pdb = os.environ.get("DEEPPBS_TEST_PDB")
    if not pdb or not os.path.exists(pdb):
        pytest.skip("Set DEEPPBS_TEST_PDB to a protein-DNA PDB to run this benchmark")

    inputs = DeepPBSSpecificityInput(pdb_paths=[pdb])
    config = DeepPBSSpecificityConfig()

    # Cold + warm to exercise persistent-worker reuse.
    for _ in range(2):
        output = run_deeppbs_specificity(inputs, config)
        validate_output(output)
        assert len(output.results) == 1
        assert all(len(row) == 4 for row in output.results[0].predicted_ppm)
