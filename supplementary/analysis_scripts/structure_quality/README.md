# Structure-quality analyses

`collect_result.py` writes one `results.csv` with four sections:

| Section | Seeds | Reported in |
|---|---|---|
| `threshold_grid` | 42 only | Table S7.1 |
| `quality_bins` | 42 only | Table S7.2 (pTM side) |
| `branch_gain` | 40-44 mean ± sd | Table S7.4 |
| `train_quality` | 40-44 mean ± sd | Table S7.3 |

Do **not** compare a seed-42 value to a 5-seed mean as if they were the same
quantity. The threshold grid and quality bins describe the fixed benchmark
model (seed 42); branch gain and training-set quality control use the five
seeds 40-44 and report sample standard deviations.

Example, same pTM bin 0.50-0.60:

- quality bin: Macro MCC **0.331** (seed 42)
- branch gain: full model **0.329 ± 0.028** (seeds 40-44)

## Regenerate

```bash
python collect_result.py \
    --val-pred <prediction_results.csv> \
    --selection <structure_selection.csv> \
    --full-root <full-model run root> \
    --esms-root <ESM-S-only run root> \
    --saprot-root <SaProt-only run root> \
    --train-quality-root <train_quality directory> \
    --out results.csv
```

The three run roots must contain `seed40` ... `seed44` subdirectories with
`prediction_results.csv`. `train_quality` must contain
`plddt{70,80,90}/seed40-44/classification_report.csv` and
`train_ptm0.5_plddt{70,80,90}.csv`.
