#!/bin/bash
# =============================================================================
# CAPOS quickstart -- direct inference from precomputed features.
#
# Runs the production classifier on ten bundled proteins using precomputed
# ESM-S/SaProt/ProtT5 features and prints the model predictions. CPU only:
# no GPU, no pretrained encoder weights, no network.
#
# The .h5 files use the same feature schema as features produced by the full
# pipeline. This is the exact classifier inference step used after
# `capos featurize` / `capos run`; no expected labels are printed.
#
# Usage:
#   ./quickstart.sh [path/to/capos-full.sif]
#
# The default image is images/capos-full.sif. Results are written to
# quickstart/out/.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 0 ]]; then
    SIF="$1"
else
    mapfile -t sifs < <(find "$ROOT/images" -maxdepth 1 -type f -name '*.sif' -print 2>/dev/null | sort)
    if [[ ${#sifs[@]} -eq 1 ]]; then
        SIF="${sifs[0]}"
    else
        SIF="$ROOT/images/capos-full.sif"
    fi
fi

DATA="$ROOT/quickstart/example_data"
OUT="$ROOT/quickstart/out"

[[ -f "$SIF" ]] || { echo "!! image not found: $SIF" >&2; exit 1; }
[[ -x "$(command -v singularity)" ]] || { echo "!! singularity is not on PATH" >&2; exit 1; }
for file in "$DATA/example_esms.h5" "$DATA/example_saprot.h5" \
            "$DATA/example_prott5.h5" "$DATA/example.fasta"; do
    [[ -f "$file" ]] || { echo "!! missing file: $file" >&2; exit 1; }
done

mkdir -p "$OUT"
rm -f "$OUT/predictions.csv" "$OUT/predictions_seq.csv"

# All paths inside the container are absolute. /data/out is the only writable
# mount, so it also serves as the working directory.
exec_in() {
    singularity exec --cleanenv --pwd /data/out \
        --bind "$DATA":/data/example_data:ro \
        --bind "$OUT":/data/out \
        "$SIF" "$@"
}

# Print a compact view of the model output; the CSV keeps the sequence and all
# per-class probabilities.
SHOW_PY=$(cat <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
print("%-10s %-10s %8s" % ("ID", "Predicted", "P(pred)"))
for row in rows:
    label = row["Predicted_Label"]
    probability = float(row.get("Prob_Class_" + label, "nan"))
    print("%-10s %-10s %8.4f" % (row["ID"], label, probability))
PY
)

echo "image: $SIF"
exec_in capos --version

echo
echo "== CAPOS (ESM-S + SaProt) =="
exec_in capos predict \
    --seq-feat /data/example_data/example_esms.h5 \
    --struct-feat /data/example_data/example_saprot.h5 \
    --weights /opt/capos/weights/capos_model.pth \
    --sequences /data/example_data/example.fasta \
    --out /data/out/predictions.csv
exec_in python3 -c "$SHOW_PY" /data/out/predictions.csv

echo
echo "== CAPOS-seq (ProtT5) =="
exec_in capos predict --model capos-seq \
    --seq-feat /data/example_data/example_prott5.h5 \
    --weights /opt/capos/weights/capos-seq_model.pth \
    --sequences /data/example_data/example.fasta \
    --out /data/out/predictions_seq.csv
exec_in python3 -c "$SHOW_PY" /data/out/predictions_seq.csv

echo
echo "Predictions written:"
echo "  $OUT/predictions.csv"
echo "  $OUT/predictions_seq.csv"
echo "Quickstart complete."
