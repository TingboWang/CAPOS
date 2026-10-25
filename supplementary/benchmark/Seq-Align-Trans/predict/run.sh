#!/bin/bash
# Run the Seq-Align-Trans MMseqs2 label-transfer baseline locally.
# Requires an mmseqs executable on PATH (or pass --mmseqs).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
exec python train.py "$@"
