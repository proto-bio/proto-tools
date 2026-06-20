#!/bin/bash
# Setup script for the NA-MPNN specificity standalone environment.
#
# NA-MPNN is not on PyPI: the tool shells out to a local NA-MPNN repository
# checkout (config.na_mpnn_repo_path) and a specificity checkpoint
# (config.checkpoint_path). Neither can be auto-downloaded — the user must
# clone https://github.com/akubaney/NA-MPNN and obtain the checkpoint, then
# point the config (or PROTO_NA_MPNN_SPECIFICITY_WEIGHTS_DIR) at them.
set -euo pipefail
source standalone_helpers.sh

echo "Setting up NA-MPNN specificity standalone environment..."

# ─── Fail-fast weights precheck ─────────────────────────────────────────────
# The NA-MPNN specificity checkpoint (s_70114.pt) is not auto-fetchable. On a
# host without it provisioned the helper emits the ASSET_NOT_AVAILABLE sentinel
# (exit 64) so the test layer skips rather than fails.
proto_resolve_asset_availability na_mpnn_specificity "*.pt" \
    "https://github.com/akubaney/NA-MPNN" \
    weights \
    "$(cat <<'HINT'
NA-MPNN and its specificity checkpoint are NOT automatically downloaded.

  1. Clone the NA-MPNN repository (https://github.com/akubaney/NA-MPNN) and
     point NAMPNNSpecificityConfig.na_mpnn_repo_path at the checkout.
  2. Obtain the specificity checkpoint (e.g. s_70114.pt) and point
     NAMPNNSpecificityConfig.checkpoint_path at it, OR place it in the
     resolved directory above (or set PROTO_NA_MPNN_SPECIFICITY_WEIGHTS_DIR).

See notes/storage.md for PROTO_MODEL_CACHE / PROTO_HOME rules.
HINT
)"

echo "Installing uv package manager..."
pip install uv

echo "Installing PyTorch: ${RECOMMENDED_TORCH_SPEC:-torch} (platform: ${DETECTED_COMPUTE_PLATFORM:-unknown})"
uv pip install "${RECOMMENDED_TORCH_SPEC:-torch}" --torch-backend=auto

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "NA-MPNN specificity setup complete."
