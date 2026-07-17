"""proto_tools/tools/structure_prediction/opendde/opendde.py.

All-atom biomolecular structure prediction using OpenDDE.

Example:
    >>> from proto_tools.tools.structure_prediction.opendde import run_opendde, OpenDDEConfig, OpenDDEInput
    >>> inputs = OpenDDEInput(complexes=[["MVLSPADKTNVKAAW", "GSSGSSGSS"]])
    >>> result = run_opendde(inputs, OpenDDEConfig())
    >>> print(f"pLDDT: {result.structures[0].metrics['avg_plddt']:.1f}")
"""

import contextlib
import json
import os
import tempfile
from collections.abc import Iterator
from logging import getLogger
from typing import Any, ClassVar, Literal

from proto_tools.entities.structures import BFactorType, Structure
from proto_tools.tools.structure_prediction.opendde.helpers import (
    build_chain_msa_paths,
    complex_to_opendde_json,
)
from proto_tools.tools.structure_prediction.shared_data_models import (
    Complex,
    ComplexMSAs,
    MSAStructurePredictionConfig,
    StructurePredictionInput,
    StructurePredictionOutput,
    normalize_output_chain_ids,
)
from proto_tools.tools.tool_registry import tool
from proto_tools.utils import ConfigField, ToolInstance
from proto_tools.utils.progress import progress_bar
from proto_tools.utils.tool_io import Metrics, MetricSpec

logger = getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================
# Input:
class OpenDDEInput(StructurePredictionInput):
    """Input object for OpenDDE structure prediction.

    Defines the input for predicting all-atom 3D structures of proteins, nucleic
    acids, and ligands (and their complexes) with OpenDDE, a co-folding model.

    Inherits from ``StructurePredictionInput``.

    Attributes:
        complexes (list[Complex]): List of complexes to predict structures for.
            Inherited from ``StructurePredictionInput``. Each complex can contain
            one or more chains of proteins, DNA, RNA, or ligands.
        msas (list[ComplexMSAs] | None): Pre-computed MSAs, one entry per complex.
            Each entry is a ``ComplexMSAs`` (per-chain MSAs keyed by chain index).
            Populated by ``preprocess()`` or supplied directly. Default: None.
    """

    # OpenDDE supports the standard biomolecular entity types.
    SUPPORTED_ENTITY_TYPES: ClassVar[set[str]] = {"protein", "dna", "rna", "ligand"}
    ALLOWS_CHAIN_MODIFICATIONS = True


class OpenDDEMetrics(Metrics):
    """Per-structure metrics emitted by OpenDDE prediction.

    Metrics documented in ``metric_spec``:
        avg_plddt (float): Mean predicted LDDT score (0-100). Always present.
        ptm (float): Predicted TM-score (0-1). Always present.
        iptm (float): Interface predicted TM-score (0-1); 0.0 for single-chain inputs.
        gpde (float): Global predicted distance error (lower is better). Always present.
        ranking_score (float): OpenDDE ranking score used to select the best sample. Always present.
        has_clash (bool): Whether the predicted structure has a steric clash. Always present.

    Note:
        OpenDDE does not emit a per-token PAE matrix, so no ``avg_pae``/``pae`` is
        reported; ``gpde`` is its global confidence-of-placement metric.
    """

    metric_spec: ClassVar[dict[str, MetricSpec]] = {
        "avg_plddt": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 100.0,
            "better_values_are": "higher",
        },
        "ptm": {"availability": "always", "type": "float", "min": 0.0, "max": 1.0, "better_values_are": "higher"},
        "iptm": {
            "availability": "always",
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "better_values_are": "higher",
        },
        "gpde": {"availability": "always", "type": "float", "min": 0.0, "max": None, "better_values_are": "lower"},
        "ranking_score": {
            "availability": "always",
            "type": "float",
            "min": None,
            "max": None,
            "better_values_are": "higher",
        },
        "has_clash": {
            "availability": "always",
            "type": "bool",
            "min": None,
            "max": None,
            "better_values_are": "lower",
        },
    }
    primary_metric: str | None = "avg_plddt"


# Output:
class OpenDDEOutput(StructurePredictionOutput):
    """OpenDDE prediction output.

    Attributes:
        structures (list[Structure]): Predicted structures, one per input complex,
            each carrying an :class:`OpenDDEMetrics` instance on ``.metrics``.
    """


# Config:
class OpenDDEConfig(MSAStructurePredictionConfig):
    """Configuration object for OpenDDE structure prediction.

    Configures OpenDDE, an all-atom co-folding model for proteins, nucleic acids,
    and ligands. Fields map to the ``opendde pred`` CLI (``--sample``, ``--step``,
    ``--cycle``, ``--use_template``, ``--use_rna_msa``, ``-n``).

    Inherits from ``MSAStructurePredictionConfig``.

    Attributes:
        name (str): Name of the folding job; drives the output path. Default: ``"opendde_job"``.
        model_name (Literal["opendde_v1", "opendde_abag"]): OpenDDE checkpoint to use
            (``-n``). ``opendde_abag`` is antibody-antigen tuned. Default: ``"opendde_v1"``.
        num_samples (int): Independent diffusion samples per complex (``--sample``);
            the best by ranking score is returned. Default: 1.
        num_steps (int): Diffusion denoising steps (``--step``). Higher = more
            refined but slower. Default: 200.
        num_cycles (int): Recycling iterations (``--cycle``). Higher = more accurate
            but slower. Default: 10.
        use_template (bool): Enable OpenDDE's template pipeline (``--use_template``).
            Default: False.
        use_rna_msa (bool): Enable OpenDDE's RNA MSA pipeline (``--use_rna_msa``).
            Default: False.
        root_dir (str | None): OpenDDE runtime/asset root (``OPENDDE_ROOT_DIR``:
            checkpoints and common assets). If ``None``, resolves from
            ``PROTO_OPENDDE_WEIGHTS_DIR`` / ``PROTO_MODEL_CACHE``. Default: None.
        load_checkpoint_path (str | None): Explicit checkpoint file override
            (``--load_checkpoint_path``); bypasses ``model_name`` resolution. Default: None.
        use_msa (bool): Auto-generate protein MSAs via MMseqs2 homology search;
            supplied MSAs always override this. Inherited. Default: True.
        pair_heterocomplex_msas (bool): Taxonomy-pair heterocomplex protein MSAs.
            Inherited. Default: True.
        msa_search_config (Mmseqs2HomologySearchConfig | None): MMseqs2 search config;
            only used when ``use_msa=True``. Inherited. Default: None.
        include_pae_matrix (bool): Inherited; ignored by OpenDDE, which emits no
            per-token PAE matrix. Default: False.
        device (str): Device to run on (``"cuda"``, ``"cpu"``). Inherited. Default: ``"cuda"``.
        timeout (int | None): Max execution time in seconds; ``None`` waits indefinitely. Default: 1200.
    """

    name: str = ConfigField(
        title="OpenDDE Job Name",
        default="opendde_job",
        description="Name of the OpenDDE folding job; drives the output path.",
    )
    model_name: Literal["opendde_v1", "opendde_abag"] = ConfigField(
        title="OpenDDE Model Name",
        default="opendde_v1",
        description="OpenDDE checkpoint (-n): 'opendde_v1' general-purpose or 'opendde_abag' antibody-antigen tuned.",
    )
    num_samples: int = ConfigField(
        title="Number of Samples",
        default=1,
        ge=1,
        description="Independent diffusion samples per complex (--sample); best by ranking score is kept.",
    )
    num_steps: int = ConfigField(
        title="Number of Diffusion Steps",
        default=200,
        ge=1,
        description="Diffusion denoising steps (--step). Higher = more refined but slower.",
    )
    num_cycles: int = ConfigField(
        title="Number of Recycling Cycles",
        default=10,
        ge=1,
        description="Recycling iterations (--cycle). Higher = more accurate but slower.",
    )
    use_template: bool = ConfigField(
        title="Use Template",
        default=False,
        description="Enable OpenDDE's template search pipeline (--use_template).",
    )
    use_rna_msa: bool = ConfigField(
        title="Use RNA MSA",
        default=False,
        description="Enable OpenDDE's RNA MSA pipeline (--use_rna_msa).",
    )
    root_dir: str | None = ConfigField(
        title="OpenDDE Root Directory",
        default=None,
        description="OpenDDE runtime/asset root (OPENDDE_ROOT_DIR). If unset, resolves from env vars.",
    )
    load_checkpoint_path: str | None = ConfigField(
        title="Checkpoint Path Override",
        default=None,
        description="Explicit checkpoint file (--load_checkpoint_path); bypasses model_name resolution.",
    )
    timeout: int | None = ConfigField(
        title="Timeout",
        default=1200,
        ge=1,
        description="Maximum execution time in seconds.",
        include_in_key=False,
    )

    def cloud_unsupported_reason(self) -> str | None:
        """A local ``root_dir`` / ``load_checkpoint_path`` is not present on a hosted worker."""
        if self.root_dir:
            return (
                "root_dir points to a local OpenDDE asset directory not available on device='cloud'. "
                "Unset it, or run locally with device='cuda'/'cpu'."
            )
        if self.load_checkpoint_path:
            return (
                "load_checkpoint_path points to a local checkpoint not available on device='cloud'. "
                "Unset it, or run locally with device='cuda'/'cpu'."
            )
        return None


# ============================================================================
# Tool Implementation
# ============================================================================
def example_input() -> Any:
    """Minimal valid input for testing and examples."""
    return OpenDDEInput(complexes=["MKTL"])  # type: ignore[list-item]


@contextlib.contextmanager
def _config_overrides_env(root_dir: str | None) -> Iterator[None]:
    """Mirror ``config.root_dir`` onto ``OPENDDE_ROOT_DIR`` for the dispatch, then restore.

    setup.sh's asset resolution and the env_vars.txt passthrough only see env
    vars, not the config. When a caller supplies ``root_dir`` it must take
    precedence over the env fallback, so mirror it for the duration of the call.

    Args:
        root_dir (str | None): Config-supplied OpenDDE root. When falsy, the env
            var is left untouched and resolution falls back to
            ``PROTO_OPENDDE_WEIGHTS_DIR`` / ``PROTO_MODEL_CACHE`` defaults.
    """
    if not root_dir:
        yield
        return
    key = "OPENDDE_ROOT_DIR"
    sentinel = object()
    original: Any = os.environ.get(key, sentinel)
    os.environ[key] = root_dir
    try:
        yield
    finally:
        if original is sentinel:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


@tool(
    key="opendde-prediction",
    label="OpenDDE Structure Prediction",
    category="structure_prediction",
    input_class=OpenDDEInput,
    config_class=OpenDDEConfig,
    output_class=OpenDDEOutput,
    metrics_class=OpenDDEMetrics,
    description="All-atom biomolecular structure prediction using OpenDDE",
    uses_gpu=True,
    device_count="1-2",
    example_input=example_input,
    iterable_input_fields=["complexes", "msas"],
    iterable_output_field="structures",
    cacheable=True,
    stochastic=True,
)
def run_opendde(inputs: OpenDDEInput, config: OpenDDEConfig, instance: Any = None) -> OpenDDEOutput:
    """Predict all-atom 3D structures using the OpenDDE co-folding model.

    Runs OpenDDE via local GPU execution in an isolated environment. Each complex
    is folded independently; the best of ``num_samples`` diffusion samples (by
    ranking score) is returned per complex. MSA generation is delegated to the
    MMseqs2 homology-search tool and supplied to OpenDDE as A3M files.

    Args:
        inputs (OpenDDEInput): Validated input with one or more complexes to fold.
        config (OpenDDEConfig): Validated OpenDDE configuration (checkpoint,
            sampling parameters, MSA settings, execution options).
        instance (Any): Optional ToolInstance for subprocess execution.

    Returns:
        OpenDDEOutput: Output with one ``Structure`` per input complex, each
            carrying :class:`OpenDDEMetrics` (avg_plddt, avg_pae, ptm, iptm, gpde,
            ranking_score) on ``.metrics``.

    Example:
        >>> inputs = OpenDDEInput(complexes=[["MVLSPADKTNVKAAW", "GSSGSSGSS"]])
        >>> config = OpenDDEConfig(num_steps=200, num_cycles=10, verbose=True)
        >>> result = run_opendde(inputs, config)
        >>> print(f"pLDDT: {result.structures[0].metrics['avg_plddt']:.1f}")
    """
    base_seed = config.seed if config.seed is not None else config.get_random_int()
    with _config_overrides_env(config.root_dir):
        results = [
            _run_opendde_on_complex(
                config=config,
                sp_complex=comp,
                complex_msas=inputs.msas[dispatch_idx] if inputs.msas else None,
                instance=instance,
                # Advance the seed per complex so duplicate inputs get distinct seeds.
                seed=base_seed + dispatch_idx,
                dispatch_idx=dispatch_idx,
            )
            for dispatch_idx, comp in enumerate(
                progress_bar(
                    inputs.complexes, desc="Folding structures (OpenDDE)", unit="complex", total=len(inputs.complexes)
                )
            )
        ]
    return OpenDDEOutput(
        structures=results,
        metadata={
            "num_complexes": len(results),
            "total_chains": sum(s.num_chains for s in results),
        },
    )


def _run_opendde_on_complex(
    config: OpenDDEConfig,
    sp_complex: Complex,
    complex_msas: ComplexMSAs | None,
    instance: Any,
    seed: int,
    dispatch_idx: int,
) -> Structure:
    """Fold a single complex with OpenDDE and return its best-ranked structure.

    Args:
        config (OpenDDEConfig): OpenDDE configuration.
        sp_complex (Complex): The complex to fold.
        complex_msas (ComplexMSAs | None): Pre-computed per-chain MSAs for this complex.
        instance (Any): Optional ToolInstance for persistent execution.
        seed (int): Per-complex ``modelSeeds`` value passed to OpenDDE.
        dispatch_idx (int): Position of this complex in the batch (used for the job name).

    Returns:
        Structure: The best-ranked predicted structure with OpenDDE metrics attached.
    """
    if config.verbose:
        logger.info("Using local GPU for OpenDDE structure prediction...")

    job_name = f"{config.name}_{dispatch_idx}"
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "opendde_output")
        os.makedirs(output_dir)

        # Honor MSAs whenever present (auto-generated when use_msa=True, or supplied
        # directly). OpenDDE consumes them via unpairedMsaPath; templates are not
        # generated by proto_tools, so its internal search stays off by default.
        chain_msa_paths = build_chain_msa_paths(sp_complex, complex_msas, temp_dir, verbose=config.verbose)
        input_json = complex_to_opendde_json(sp_complex.chains, job_name, [seed], chain_msa_paths=chain_msa_paths)

        input_json_path = os.path.join(temp_dir, f"{job_name}.json")
        with open(input_json_path, "w") as f:
            json.dump([input_json], f, indent=2)

        # OpenDDE runs its own MSA search only when no A3M paths are supplied and
        # the caller asked for it; supplied paths always take precedence.
        run_internal_msa = config.use_msa and not chain_msa_paths

        input_data = {
            "operation": "predict",
            "input_json_path": input_json_path,
            "output_dir": output_dir,
            "job_name": job_name,
            "model_name": config.model_name,
            "load_checkpoint_path": config.load_checkpoint_path,
            "num_samples": config.num_samples,
            "num_steps": config.num_steps,
            "num_cycles": config.num_cycles,
            "use_msa": run_internal_msa,
            "use_template": config.use_template,
            "use_rna_msa": config.use_rna_msa,
            "seed": seed,
            "device": config.device,
            "verbose": config.verbose,
        }

        output_data = ToolInstance.dispatch("opendde", input_data, instance=instance, config=config)

    cif_output = output_data["structure_cif_output"]
    formatted_metrics = output_data["metrics"]

    metrics_dict: dict[str, Any] = {
        "avg_plddt": float(formatted_metrics["avg_plddt"]),
        "ptm": float(formatted_metrics["ptm"]),
        "iptm": float(formatted_metrics["iptm"]),
        "gpde": float(formatted_metrics["gpde"]),
        "ranking_score": float(formatted_metrics["ranking_score"]),
        "has_clash": bool(formatted_metrics["has_clash"]),
    }

    structure = Structure(
        structure=cif_output,
        b_factor_type=BFactorType.PLDDT,
        metrics=OpenDDEMetrics(**metrics_dict),
        source="opendde-prediction",
    )
    return normalize_output_chain_ids(structure, sp_complex.chains)
