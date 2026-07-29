"""Local ESM C sparse autoencoder (SAE) feature extraction.

Loads an ESM C backbone through Transformers and attaches the sparse
autoencoders trained against the requested layers. Shares the ``biohub_esm``
env with the ESM C and ESM3 wrappers.
"""

import json
import sys
from typing import Any

import torch
from standalone_helpers import get_logger, serialize_output
from tqdm import tqdm

logger = get_logger(__name__)

# HuggingFace repo holding the Transformers-format backbone weights. These are
# distinct from the `esm`-package repos the esmc wrapper loads.
BACKBONE_REPOS = {
    "esmc_300m": "biohub/ESMC-300M",
    "esmc_600m": "biohub/ESMC-600M",
    "esmc_6b": "biohub/ESMC-6B",
}


class ESMCSAEModel:
    """ESM C backbone with sparse autoencoders attached to selected layers."""

    def __init__(self, model_checkpoint: str = "esmc_300m", sae_repo: str = "", layers: tuple[int, ...] = ()):
        """Initialize the wrapper.

        Args:
            model_checkpoint: ESM C backbone key, e.g. ``"esmc_300m"``.
            sae_repo: HuggingFace repo id holding the SAE weights.
            layers: Backbone layer indices to attach SAEs to.
        """
        self._loaded = False
        self.model_checkpoint = model_checkpoint
        self.sae_repo = sae_repo
        self.layers = tuple(layers)
        self.device: str | None = None
        self.model: Any = None
        self.tokenizer: Any = None

    def __call__(
        self,
        sequences: list[str],
        batch_size: int = 1,
        device: str = "cuda",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Extract SAE features for each sequence.

        Args:
            sequences: Protein sequences.
            batch_size: Sequences per forward pass.
            device: Device to run on.
            verbose: Whether to show progress.

        Returns:
            Dictionary with a ``features`` list holding one entry per input
            sequence, keyed by layer index, each with ``indices`` and
            ``magnitudes`` arrays of shape (residues, k).
        """
        if not self._loaded:
            self.load(device, verbose)
        elif self.device != device:
            self.to_device(device)

        if not sequences:
            raise ValueError("esmc_sae: __call__ requires at least one sequence")
        if any(len(seq) == 0 for seq in sequences):
            raise ValueError("esmc_sae: __call__ does not support empty sequences")

        batches = [sequences[i : i + batch_size] for i in range(0, len(sequences), batch_size)]
        features: list[dict[str, dict[str, list[list[float]]]]] = []

        for batch in tqdm(
            batches, desc="ESM C SAE inference", unit="batch", total=len(batches), disable=not verbose
        ):
            inputs = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)

            # The SAE stack emits one row per *unpadded* token, concatenated
            # across the batch, so rows must be split by each sequence's real
            # token count rather than reshaped to (batch, padded_len, codebook).
            token_counts = inputs["attention_mask"].sum(dim=1).tolist()
            features.extend(self._split_batch(outputs["sae_outputs"], token_counts))

        return {"features": features}

    def _split_batch(
        self, sae_outputs: dict[str, torch.Tensor], token_counts: list[int]
    ) -> list[dict[str, dict[str, list[list[float]]]]]:
        """Split concatenated SAE rows into per-sequence top-k features.

        Args:
            sae_outputs: Mapping of ``layer{N}`` to a sparse (total_tokens, codebook)
                tensor covering every unpadded token in the batch.
            token_counts: Real token count per sequence, including start/end tokens.

        Returns:
            One dict per sequence, keyed by layer index as a string.
        """
        per_sequence: list[dict[str, dict[str, list[list[float]]]]] = [{} for _ in token_counts]

        for layer in self.layers:
            dense = sae_outputs[f"layer{layer}"]
            if dense.is_sparse:
                dense = dense.to_dense()

            offset = 0
            for seq_idx, count in enumerate(token_counts):
                # Strip the start and end tokens to align with the input residues.
                rows = dense[offset + 1 : offset + count - 1]
                offset += count

                magnitudes, indices = torch.sort(rows, dim=-1, descending=True)
                active = (rows != 0).sum(dim=-1)
                keep = int(active.max().item()) if active.numel() else 0

                per_sequence[seq_idx][str(layer)] = {
                    "indices": serialize_output(indices[:, :keep].to(torch.int64)),
                    "magnitudes": serialize_output(magnitudes[:, :keep].to(torch.float32)),
                }

        return per_sequence

    # ============================================================================
    # Helper Functions
    # ============================================================================
    def load(self, device: str, verbose: bool = False) -> None:
        """Load the backbone and SAE layers onto the device."""
        from transformers import AutoModel, AutoTokenizer
        from transformers.models.esmc.modeling_esmc_sae import ESMCSAEModel as HFSAEModel

        repo = BACKBONE_REPOS[self.model_checkpoint]
        if verbose:
            logger.info(f"Loading ESM C backbone {repo} on {device}")
        logger.update_status(f"Loading {self.model_checkpoint}")

        self.model = AutoModel.from_pretrained(repo, device_map=device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(repo)

        # Fetch only the requested layers; the all-layer SAE repos hold one file
        # per backbone layer and the 6B collection is tens of gigabytes.
        allow_patterns = ["config.json"] + [f"layer_{layer}.safetensors" for layer in self.layers]
        logger.update_status(f"Loading SAE layers {list(self.layers)}")
        sae = HFSAEModel.from_pretrained(self.sae_repo, allow_patterns=allow_patterns, device=self.model.device)
        sae.initialize_layers(list(self.layers))
        self.model.add_sae_models([sae.layers[str(layer)] for layer in self.layers])

        self.device = device
        self._loaded = True

        if verbose:
            logger.info(f"ESM C SAE model loaded ({self.sae_repo}, layers {list(self.layers)})")

    def to_device(self, device: str) -> None:
        """Move the model to a different device."""
        from standalone_helpers import move_model_to_device

        if not self._loaded:
            raise ValueError("esmc_sae: cannot move unloaded model to device — call load() first")

        if self.device != device:
            self.model = move_model_to_device(self.model, self.device, device)
            self.device = device

    def unload(self, verbose: bool = False) -> None:
        """Move the model to CPU to free GPU memory."""
        if self._loaded and self.device != "cpu":
            if verbose:
                logger.info(f"Unloading {self.__class__.__name__} from GPU")

            self.model = self.model.to("cpu")
            self.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================================================================
# Dispatch
# ============================================================================
_model: ESMCSAEModel | None = None


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    global _model
    layers = tuple(input_dict["layers"])
    if _model is None or _model.sae_repo != input_dict["sae_repo"] or _model.layers != layers:
        _model = ESMCSAEModel(
            model_checkpoint=input_dict["model_checkpoint"],
            sae_repo=input_dict["sae_repo"],
            layers=layers,
        )

    operation = input_dict["operation"]
    if operation == "sae_features":
        return _model(
            sequences=input_dict["sequences"],
            batch_size=input_dict["batch_size"],
            device=input_dict["device"],
            verbose=input_dict["verbose"],
        )
    raise ValueError(f"esmc_sae: unknown operation {operation!r}; valid: ['sae_features']")


def to_device(device: str) -> dict[str, Any]:
    """Move model to specified device (called by DeviceManager)."""
    global _model
    if _model is not None and _model._loaded:
        _model.to_device(device)
        return {"success": True, "device": device}
    return {"success": True, "device": device, "note": "model not loaded yet"}


def get_memory_stats() -> dict[str, Any]:
    """Return PyTorch memory stats for the model's device (used by ToolPool)."""
    from standalone_helpers import get_pytorch_memory_stats

    device = _model.device if _model and hasattr(_model, "device") else 0
    return get_pytorch_memory_stats(device)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("esmc_sae: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
