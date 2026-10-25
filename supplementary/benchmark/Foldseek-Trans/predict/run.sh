#!/bin/bash
# Run the Foldseek-Trans structural-transfer baseline locally.
# Prepare structures first (prepare_structures.py) and ensure a Foldseek
# executable is available (or pass --foldseek).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
exec python train.py "$@"
