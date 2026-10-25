#!/bin/bash
# =============================================================================
# CAPOS quickstart -- the out-of-the-box example.
#
#     cd quickstart && ./run_quickstart.sh
#
# Self-contained: needs nothing outside this repository. No GPU, no downloads,
# no Chai-1, no torchdrug. If `capos` is installed (pip install ..), this runs
# in seconds to a couple of minutes on a laptop CPU.
#
# What it does
#   1. runs `capos predict` on the ten bundled proteins (pre-computed features)
#   2. writes predictions.csv into quickstart/run_out/
#   3. diffs the predictions against the bundled reference (quickstart/reference_out/)
#   4. runs the sequence-only CAPOS-seq checkpoint on the same ten proteins
#
# Expected result: every predicted label matches the reference, and the
# probabilities agree to ~1e-5 (floating-point noise across BLAS builds).
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DATA="$HERE/example_data"
REF="$HERE/reference_out"
OUT="$HERE/run_out"

# The shipped checkpoints (dG=5 training split, seed 42) live in weights/.
WEIGHTS="$ROOT/weights/capos_model.pth"
WEIGHTS_SEQ="$ROOT/weights/capos-seq_model.pth"

mkdir -p "$OUT"

if ! python -c "import capos" 2>/dev/null; then
    echo "!! the 'capos' package is not installed."
    echo "   run:  pip install $ROOT"
    echo "   (or:  conda env create -f $ROOT/environment.yml && conda activate capos"
    echo "         && pip install $ROOT)"
    exit 1
fi

for f in "$DATA/example_esms.h5" "$DATA/example_saprot.h5" "$WEIGHTS"; do
    if [ ! -f "$f" ]; then
        echo "!! missing bundled file: $f"
        exit 1
    fi
done

echo "==== [1/4] CAPOS, light tier (CPU, 10 proteins) ===="
capos predict \
    --seq-feat    "$DATA/example_esms.h5" \
    --struct-feat "$DATA/example_saprot.h5" \
    --weights     "$WEIGHTS" \
    --sequences   "$DATA/example.fasta" \
    --out         "$OUT/predictions.csv"

echo
echo "==== [2/4] results ===="
column -s, -t "$OUT/predictions.csv" | cut -c1-110
echo
echo "--- reference (quickstart/reference_out/predictions.csv) ---"
column -s, -t "$REF/predictions.csv" | cut -c1-110
echo
echo "==== [3/4] checking against the bundled reference ===="
python - "$OUT/predictions.csv" "$REF/predictions.csv" <<'PY'
import csv, sys

got, ref = sys.argv[1], sys.argv[2]
g = list(csv.DictReader(open(got, newline="", encoding="utf-8")))
r = list(csv.DictReader(open(ref, newline="", encoding="utf-8")))
assert len(g) == len(r), "row count %d != reference %d" % (len(g), len(r))

bad_lab, max_dp = [], 0.0
for a, b in zip(g, r):
    assert a["ID"] == b["ID"], "ID order differs: %s vs %s" % (a["ID"], b["ID"])
    if a["Predicted_Label"] != b["Predicted_Label"]:
        bad_lab.append("%s: got %s, reference %s" % (a["ID"], a["Predicted_Label"], b["Predicted_Label"]))
    for k in a:
        if k.startswith("Prob_Class_"):
            max_dp = max(max_dp, abs(float(a[k]) - float(b[k])))

if bad_lab:
    print("FAIL: predicted labels differ from the reference:")
    for line in bad_lab:
        print("  ", line)
    sys.exit(1)

print("OK  %d/%d labels match the reference; max |dP| = %.2e"
      % (len(g) - len(bad_lab), len(g), max_dp))
PY

echo
echo "==== [4/4] CAPOS-seq (ProtT5 features) ===="
# CAPOS-seq takes ProtT5 embeddings (1024-d) rather than ESM-S (1280-d),
# so it reads a different feature file.
if [ -f "$DATA/example_prott5.h5" ] && [ -f "$WEIGHTS_SEQ" ]; then
    capos predict --model capos-seq \
        --seq-feat   "$DATA/example_prott5.h5" \
        --weights    "$WEIGHTS_SEQ" \
        --sequences  "$DATA/example.fasta" \
        --out        "$OUT/predictions_seq.csv"
    echo "--- CAPOS-seq (ID, Predicted_Label) ---"
    cut -d, -f1,3 "$OUT/predictions_seq.csv"
else
    echo "(skipped -- example_prott5.h5 or weights/capos-seq_model.pth not present)"
fi

echo
echo "Quickstart outputs in: $OUT"
echo "Quickstart is working."
