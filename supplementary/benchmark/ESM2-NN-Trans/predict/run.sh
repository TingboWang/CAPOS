#!/bin/bash
# Run the ESM2-NN-Trans nearest-neighbour baseline locally.
# Pass --features /path/to/esm2_features_all.npz when using your own embeddings.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
exec python train.py "$@"
