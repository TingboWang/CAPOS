#!/bin/bash
# Run the Random prior-sampling baseline locally.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
exec python train.py "$@"
