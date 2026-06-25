"""Tests for Metal3D tool models and registration."""

from pathlib import Path

from proto_tools import Metal3DPredictionConfig, Metal3DPredictionInput
from proto_tools.tools.tool_registry import ToolRegistry

EXAMPLE_PDB = Path(__file__).parent.parent / "dummy_data" / "pdl1.pdb"


def test_metal3d_prediction_is_registered() -> None:
    spec = ToolRegistry.get("metal3d-prediction")
    assert spec.key == "metal3d-prediction"
    assert spec.uses_gpu is True
    assert spec.config_model is Metal3DPredictionConfig


def test_metal3d_input_accepts_single_structure_with_candidate_residues() -> None:
    inputs = Metal3DPredictionInput(
        inputs={
            "structure": str(EXAMPLE_PDB),
            "candidate_residues": {"A": [1, 2, 3]},
        }
    )

    assert len(inputs.inputs) == 1
    assert inputs.inputs[0].candidate_residues is not None
    assert inputs.inputs[0].candidate_residues.chains == {"A": [1, 2, 3]}
