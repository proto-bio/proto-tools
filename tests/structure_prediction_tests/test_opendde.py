"""tests/structure_prediction_tests/test_opendde.py.

Tests for OpenDDE all-atom structure prediction (``opendde-prediction``).
"""

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

from proto_tools.entities.ligands import Fragment
from proto_tools.entities.structures import is_valid_structure
from proto_tools.tools import (
    Chain,
    OpenDDEConfig,
    OpenDDEInput,
    run_opendde,
)
from proto_tools.tools.structure_prediction.opendde.helpers import complex_to_opendde_json
from tests._structure_fixtures import synthetic_cif
from tests.conftest import benchmark_twice
from tests.structure_prediction_tests._fasta_helpers import load_benchmark_complex
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.test_export_functionality import validate_export_output

# Cro repressor from bacteriophage lambda — short, well-folded test protein.
_CRO_SEQUENCE = "MQTQNNSREKQAAALERLFLSCFLKDPVPKPLQEGTCDDVLCRELLNESETHLVQSIFRKESKVPGA"
# A short, foldable peptide for GPU smoke tests.
_TINY_PEPTIDE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"


# ── OpenDDE JSON shape: entity mapping ───────────────────────────────────────


def test_opendde_json_top_level_and_protein_entry():
    """The job dict carries name/modelSeeds/sequences and a protein → proteinChain entry."""
    chains = [Chain(sequence="MVLSPADKTN", entity_type="protein")]
    job = complex_to_opendde_json(chains, "myjob", [7])

    assert set(job) == {"name", "modelSeeds", "sequences"}
    assert job["name"] == "myjob"
    assert job["modelSeeds"] == [7]

    [entry] = job["sequences"]
    assert set(entry) == {"proteinChain"}
    protein = entry["proteinChain"]
    assert protein["sequence"] == "MVLSPADKTN"
    assert protein["count"] == 1
    assert protein["id"] == ["A"]
    # No MSA supplied → single-sequence, no unpairedMsaPath key.
    assert "unpairedMsaPath" not in protein


def test_opendde_ligand_uses_ccd_code_when_available():
    """A Fragment with a resolved ccd_code serializes to CCD_-prefixed ``ligand`` (CCD preferred).

    OpenDDE parses a ligand string as SMILES unless it carries the ``CCD_`` prefix,
    so bare CCD codes must be prefixed.
    """
    atp = Fragment(ccd_code="ATP")
    assert atp.ccd_code == "ATP"  # invariant guard

    job = complex_to_opendde_json([Chain(sequence="MKTLPGCDA", entity_type="protein"), atp], "j", [0])
    ligand_entries = [e["ligand"] for e in job["sequences"] if "ligand" in e]
    assert ligand_entries == [{"ligand": "CCD_ATP", "count": 1, "id": ["B"]}]


def test_opendde_ligand_falls_back_to_smiles_when_no_ccd_match():
    """A ligand with no CCD match serializes with its raw SMILES in the ``ligand`` field."""
    # Synthetic perfluorinated terphenyl — not in the wwPDB CCD database.
    novel_smiles = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)c1ccc(-c2ccc(-c3ccccc3)cc2)cc1"
    novel = Fragment(smiles=novel_smiles)
    assert novel.ccd_code is None  # invariant guard

    job = complex_to_opendde_json([Chain(sequence="MKTLPGCDA", entity_type="protein"), novel], "j", [0])
    [ligand] = [e["ligand"] for e in job["sequences"] if "ligand" in e]
    assert ligand == {"ligand": novel_smiles, "count": 1, "id": ["B"]}


def test_opendde_dna_maps_to_dna_sequence():
    """A DNA chain maps to a ``dnaSequence`` entry carrying the raw nucleotide sequence."""
    chains = [Chain(sequence="ACGTACGT", entity_type="dna")]
    job = complex_to_opendde_json(chains, "j", [0])

    [entry] = job["sequences"]
    assert set(entry) == {"dnaSequence"}
    assert entry["dnaSequence"] == {"sequence": "ACGTACGT", "count": 1, "id": ["A"]}


def test_opendde_chain_msa_paths_set_unpaired_msa_path_on_protein():
    """A supplied chain_msa_paths entry populates ``unpairedMsaPath`` on the protein entry."""
    chains = [Chain(sequence="MVLSPADKTN", entity_type="protein")]
    job = complex_to_opendde_json(chains, "j", [0], chain_msa_paths={"A": "/scratch/chain_A.a3m"})

    assert job["sequences"][0]["proteinChain"]["unpairedMsaPath"] == "/scratch/chain_A.a3m"


def test_opendde_protein_modifications_prefixed_ptm():
    """Protein PTMs serialize as CCD_-prefixed ptmType / ptmPosition."""
    chains = [Chain(sequence="MVLSPADKTN", entity_type="protein", modifications=[(4, "SEP")])]
    job = complex_to_opendde_json(chains, "j", [0])

    mods = job["sequences"][0]["proteinChain"]["modifications"]
    assert mods == [{"ptmType": "CCD_SEP", "ptmPosition": 4}]


def test_opendde_dna_modifications_prefixed_base():
    """Nucleic-acid modifications serialize as CCD_-prefixed modificationType / basePosition."""
    chains = [Chain(sequence="ACGT", entity_type="dna", modifications=[(2, "5CM")])]
    job = complex_to_opendde_json(chains, "j", [0])

    mods = job["sequences"][0]["dnaSequence"]["modifications"]
    assert mods == [{"modificationType": "CCD_5CM", "basePosition": 2}]


# ── Checkpoint selection (standalone argv) ───────────────────────────────────


def _load_standalone_inference():
    """Import the standalone ``inference.py`` with a stubbed ``standalone_helpers``."""
    _sh = sys.modules.setdefault("standalone_helpers", types.SimpleNamespace())
    if not hasattr(_sh, "get_logger"):
        _sh.get_logger = logging.getLogger
    standalone_dir = (
        Path(__file__).resolve().parents[2]
        / "proto_tools"
        / "tools"
        / "structure_prediction"
        / "opendde"
        / "standalone"
    )
    spec = importlib.util.spec_from_file_location("_opendde_si_for_tests", standalone_dir / "inference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The console script only exists inside the tool venv; argv assembly is what is under test.
    module._resolve_opendde_bin = lambda: "/venv/bin/opendde"
    return module


def _build_cmd(model, **overrides):
    """Call ``OpenDDEModel._build_cmd`` with defaults for the args under test."""
    kwargs = {
        "input_json_path": "input.json",
        "output_dir": "output",
        "job_name": "job",
        "model_name": "opendde_v1",
        "load_checkpoint_path": None,
        "num_samples": 1,
        "num_steps": 200,
        "num_cycles": 10,
        "use_msa": False,
        "use_template": False,
        "use_rna_msa": False,
        "seed": 0,
        "device": "cuda",
    }
    kwargs.update(overrides)
    return model._build_cmd(**kwargs)


def test_opendde_model_name_never_reaches_the_n_flag(tmp_path):
    """Upstream's ``-n`` accepts only ``opendde_v1``; ``opendde_abag`` routes via checkpoint path."""
    mod = _load_standalone_inference()
    model = mod.OpenDDEModel()
    model.root_dir = str(tmp_path)
    model._loaded = True
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    (ckpt_dir / "opendde_abag.pt").write_bytes(b"")

    cmd = _build_cmd(model, model_name="opendde_abag")

    assert cmd[cmd.index("-n") + 1] == "opendde_v1"
    assert "opendde_abag" not in cmd[cmd.index("-n") + 1]
    assert cmd[cmd.index("--load_checkpoint_path") + 1] == str(ckpt_dir / "opendde_abag.pt")


def test_opendde_default_model_sends_no_checkpoint_override(tmp_path):
    """``opendde_v1`` lets OpenDDE load its own default weights."""
    mod = _load_standalone_inference()
    model = mod.OpenDDEModel()
    model.root_dir = str(tmp_path)
    model._loaded = True

    assert "--load_checkpoint_path" not in _build_cmd(model, model_name="opendde_v1")


def test_opendde_missing_abag_checkpoint_fails_before_dispatch(tmp_path):
    """A derived abag path that does not exist must fail here, not inside the CLI subprocess."""
    mod = _load_standalone_inference()
    model = mod.OpenDDEModel()
    model.root_dir = str(tmp_path)
    model._loaded = True
    (tmp_path / "checkpoint").mkdir()

    with pytest.raises(FileNotFoundError, match="antibody-antigen checkpoint not found"):
        _build_cmd(model, model_name="opendde_abag")


def test_opendde_explicit_checkpoint_path_wins_over_model_name(tmp_path):
    """An explicit ``load_checkpoint_path`` is passed through untouched and skips derivation."""
    mod = _load_standalone_inference()
    model = mod.OpenDDEModel()
    model.root_dir = str(tmp_path)
    model._loaded = True

    cmd = _build_cmd(model, model_name="opendde_abag", load_checkpoint_path="/custom/ckpt.pt")

    assert cmd[cmd.index("--load_checkpoint_path") + 1] == "/custom/ckpt.pt"


# ── Config: cloud support gate ───────────────────────────────────────────────


def test_opendde_cloud_unsupported_reason():
    """A local root_dir or load_checkpoint_path yields a reason; otherwise None."""
    assert OpenDDEConfig(use_msa=False).cloud_unsupported_reason() is None
    assert OpenDDEConfig(use_msa=False, root_dir="/local/assets").cloud_unsupported_reason() is not None
    assert OpenDDEConfig(use_msa=False, load_checkpoint_path="/local/ckpt.pt").cloud_unsupported_reason() is not None


def test_opendde_config_rejects_include_pae_matrix():
    """OpenDDE emits no PAE at all; the inherited toggle must be rejected, not ignored."""
    with pytest.raises(ValueError, match="include_pae_matrix"):
        OpenDDEConfig(use_msa=False, include_pae_matrix=True)
    # The default (False) still constructs.
    assert OpenDDEConfig(use_msa=False).include_pae_matrix is False


# ── Dispatch / metric assembly (mocked worker) ───────────────────────────────


def _fake_dispatch_factory(captured, *, metrics):
    """Build a ToolInstance.dispatch stand-in that records each call into ``captured``."""

    def fake_dispatch(toolkit, input_data, **kwargs):
        captured.setdefault("toolkits", []).append(toolkit)
        captured.setdefault("input_data", []).append(input_data)
        return {"structure_cif_output": synthetic_cif(["A"]), "metrics": dict(metrics)}

    return fake_dispatch


def test_opendde_run_builds_structure_with_metrics(monkeypatch):
    """run_opendde threads config through dispatch and assembles OpenDDEMetrics (monomer iptm=0.0)."""
    captured: dict = {}
    metrics = {
        "avg_plddt": 91.2,
        "ptm": 0.85,
        "iptm": 0.0,
        "gpde": 1.2,
        "ranking_score": 0.77,
        "has_clash": False,
    }
    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.opendde.opendde.ToolInstance.dispatch",
        _fake_dispatch_factory(captured, metrics=metrics),
    )

    result = run_opendde(
        OpenDDEInput(complexes=[_CRO_SEQUENCE]),
        OpenDDEConfig(
            use_msa=False,
            num_samples=2,
            num_steps=33,
            num_cycles=4,
            model_name="opendde_abag",
        ),
    )

    assert result.success
    assert result.tool_id == "opendde-prediction"
    assert len(result.structures) == 1

    structure = result.structures[0]
    assert structure.source == "opendde-prediction"
    assert is_valid_structure(structure.structure_cif)

    m = structure.metrics
    assert m["avg_plddt"] == pytest.approx(91.2)
    assert m["ptm"] == pytest.approx(0.85)
    assert m["gpde"] == pytest.approx(1.2)
    assert m["ranking_score"] == pytest.approx(0.77)
    assert m["has_clash"] is False
    # OpenDDE always reports iptm (0.0 for a single-chain input).
    assert m["iptm"] == pytest.approx(0.0)
    assert_metrics_in_spec(result)

    # Config sampling params must reach the worker payload, not just the config object.
    assert captured["toolkits"] == ["opendde"]
    [input_data] = captured["input_data"]
    assert input_data["operation"] == "predict"
    assert input_data["num_samples"] == 2
    assert input_data["num_steps"] == 33
    assert input_data["num_cycles"] == 4
    assert input_data["model_name"] == "opendde_abag"


def test_opendde_iptm_surfaced_when_present(monkeypatch):
    """A non-None iptm from the worker is carried onto OpenDDEMetrics."""
    captured: dict = {}
    metrics = {
        "avg_plddt": 80.0,
        "ptm": 0.7,
        "iptm": 0.62,
        "gpde": 2.0,
        "ranking_score": 0.5,
        "has_clash": False,
    }
    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.opendde.opendde.ToolInstance.dispatch",
        _fake_dispatch_factory(captured, metrics=metrics),
    )

    result = run_opendde(OpenDDEInput(complexes=[_CRO_SEQUENCE]), OpenDDEConfig(use_msa=False))

    assert result.structures[0].metrics["iptm"] == pytest.approx(0.62)


def test_opendde_one_structure_per_complex(monkeypatch):
    """Each input complex yields exactly one structure (1:1 cardinality)."""
    captured: dict = {}
    metrics = {
        "avg_plddt": 88.0,
        "ptm": 0.8,
        "iptm": 0.0,
        "gpde": 1.5,
        "ranking_score": 0.6,
        "has_clash": False,
    }
    monkeypatch.setattr(
        "proto_tools.tools.structure_prediction.opendde.opendde.ToolInstance.dispatch",
        _fake_dispatch_factory(captured, metrics=metrics),
    )

    result = run_opendde(
        OpenDDEInput(complexes=[_CRO_SEQUENCE, _TINY_PEPTIDE]),
        OpenDDEConfig(use_msa=False),
    )

    assert result.success
    assert len(result.structures) == 2
    assert captured["toolkits"] == ["opendde", "opendde"]


# ── GPU tests ────────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_opendde_basic_execution():
    """Fold a tiny peptide end-to-end (MSA-free); headline pLDDT is populated.

    Skips on hosts without OpenDDE weights / a GPU (the tool env fails to build).
    """
    result = run_opendde(
        OpenDDEInput(complexes=[_TINY_PEPTIDE]),
        OpenDDEConfig(use_msa=False, num_samples=1, num_steps=50, num_cycles=3, seed=42),
    )

    assert result.success
    assert result.tool_id == "opendde-prediction"
    assert len(result.structures) == 1
    structure = result.structures[0]
    assert is_valid_structure(structure.structure_cif)
    assert structure.metrics["avg_plddt"] is not None
    assert_metrics_in_spec(result)


@pytest.mark.uses_gpu
def test_opendde_export(tmp_path):
    """A folded OpenDDE output exports a non-empty structure file to disk."""
    result = run_opendde(
        OpenDDEInput(complexes=[_TINY_PEPTIDE]),
        OpenDDEConfig(use_msa=False, num_samples=1, num_steps=50, num_cycles=3, seed=42),
    )
    assert result.success

    result.export(name="opendde_pred", export_path=str(tmp_path), file_format="cif")
    assert validate_export_output(tmp_path / "opendde_pred")


# ── Benchmark ────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("opendde-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_opendde_benchmark(request):
    """Benchmark opendde-prediction on the MfnG protein + L-tyrosine ligand (cold + warm).

    Single ~390-residue protein-ligand complex without MSA — a representative
    co-folding workload shared with the other AF3-family predictors. Cold pass
    measures weight load + first inference; warm pass measures inference only.
    """
    complex_ = load_benchmark_complex("MfnG_and_ligand")
    inputs = OpenDDEInput(complexes=[complex_])
    config = OpenDDEConfig(use_msa=False, verbose=True)

    result = benchmark_twice(request, "opendde", lambda: run_opendde(inputs=inputs, config=config))

    assert result.success, "OpenDDE benchmark run failed"
    assert len(result.structures) == 1
    assert is_valid_structure(result.structures[0].structure_cif)
    assert_metrics_in_spec(result)
