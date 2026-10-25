# CAPOS quickstart

This directory contains the ten-protein fixture used by the quickstart and the
source-level runner.

| Runner | Requirements | Output |
|---|---|---|
| `./quickstart.sh` at the repository root | Singularity CE and `images/capos-full.sif` | `quickstart/out/` |
| `./quickstart/run_quickstart.sh` | an installed `capos` package | `quickstart/run_out/` |

The image-based runner applies the packaged CAPOS and CAPOS-seq classifiers. The
source-level runner performs the same predictions from a local installation and
compares the CAPOS predictions with `reference_out/predictions.csv`.

## Contents

| File | Description |
|---|---|
| `run_quickstart.sh` | source-level runner |
| `example_data/example_esms.h5` | precomputed ESM-S features, 10 × 1280 |
| `example_data/example_saprot.h5` | precomputed SaProt/3Di features, 10 × 1280 |
| `example_data/example_prott5.h5` | precomputed ProtT5 features, 10 × 1024 |
| `example_data/example.fasta` | the ten protein sequences |
| `example_data/example_truth.csv` | reference labels |
| `reference_out/predictions.csv` | reference CAPOS predictions |
| `reference_out/predictions_seq.csv` | reference CAPOS-seq predictions |
| `make_example_data.py` | construction of the feature subset; not needed to run the quickstart |

`example_data/` is a small inference fixture, not training or benchmark data.
CAPOS takes ESM-S and SaProt features (1280 dimensions each), whereas CAPOS-seq
takes ProtT5 features (1024 dimensions). Feeding a checkpoint a feature matrix
of the wrong width causes a shape mismatch.

The ten proteins were not used for training. The quickstart verifies that the
packaged checkpoints, feature schema, and inference path are consistent. It is
not a benchmark and does not report accuracy.

## Run by hand

The following commands are equivalent to `run_quickstart.sh`:

```bash
mkdir -p run_out

# CAPOS (needs both feature files)
capos predict \
    --seq-feat    example_data/example_esms.h5 \
    --struct-feat example_data/example_saprot.h5 \
    --weights     ../weights/capos_model.pth \
    --sequences   example_data/example.fasta \
    --out         run_out/predictions.csv

# CAPOS-seq (ProtT5 only)
capos predict --model capos-seq \
    --seq-feat   example_data/example_prott5.h5 \
    --weights    ../weights/capos-seq_model.pth \
    --sequences  example_data/example.fasta \
    --out        run_out/predictions_seq.csv
```

The CAPOS-seq weights are in `../weights/capos-seq_model.pth`. The commands
assume that `quickstart/` is the current directory and that `capos` is
installed.
