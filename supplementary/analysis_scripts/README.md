# CAPOS analysis scripts

This package contains the Python scripts used for the ablation and analysis
results reported in the CAPOS main text and Supplementary Materials. Each
subfolder contains one `results.csv` summarizing that analysis; the matching
`collect_result.py` regenerates it from the original per-seed reports,
prediction files, or per-protein tables. Small input tables needed by the
scripts are kept alongside.

| Folder | Analysis | Result file | Reported in |
|---|---|---|---|
| `ablation/` | Fusion-operator and SaProt-channel ablations | `results.csv` | Figure 4; Notes S2 and S3 |
| `confusion/` | Row-normalised confusion matrices with class counts | `results.csv` | Figure S1; Results and Discussion |
| `cross_source/` | Train-on-one-source, validate-on-the-other transfer | `results.csv` | Note S7.7; Table S7.10 |
| `discarded/` | Performance on proteins excluded by the structure-quality gate | `results.csv` | Note S7.6; Tables S7.7-S7.8 |
| `homology_stratified/` | Performance stratified by training-set structural similarity | `results.csv` | Note S6.2; Table S6.2 |
| `real_structure/` | Predicted-versus-experimental structure 2x2 experiment | `results.csv` | Note S7.5; Tables S7.5-S7.6 |
| `source_stratified/` | PDB- versus UniProt-derived validation performance | `results.csv` | Note S7.7; Table S7.9 |
| `split_identity/` | Nearest-neighbour sequence identity between splits | `results.csv` | Figure 2D; Note S6.1; Table S6.1 |
| `structure_quality/` | Quality-threshold grid, quality bins, branch gain, and training-set quality control | `results.csv` | Note S7.1-S7.4; Figure S2A-B; Tables S7.1-S7.4 |
| `threshold_sweep/` | Validation-set dG threshold sweep | `results.csv` | Figure 2B; Note S1.4; Table S1.4 |
| `tm_score/` | Performance across Chai-1 versus experimental TM-score bins | `results.csv` | Note S7.2; Figure S2C; Table S7.2 |

## Result scripts

`collect_result.py` in each folder writes the folder's `results.csv` directly.
It reads only the inputs named on its command line and uses the same metric
definitions as the CAPOS benchmark: Macro MCC and Macro F1 average over all
nine classes, and a degenerate class contributes zero. Single-seed analyses
use the frozen seed-42 model; five-seed analyses report mean and sample
standard deviation over seeds 40-44.

`common.py` contains the shared metric and prediction-file helpers used by all
`collect_result.py` scripts. `threshold_sweep/collect_result.py` is the
threshold-sweep equivalent; it reads per-seed `classification_report.csv`
files and writes `results.csv`.

Data-generation scripts are kept only where they are needed to recreate the
inputs of a summary: SaProt channel extraction, ablation training, cross-source
subset construction, excluded-protein staging, 2x2 chain/feature preparation
and scoring, identity search, and TM-score calculation. They also take their
paths on the command line or from optional repository-relative defaults.

## Running the scripts

Summary scripts are run with Python 3.10+ and NumPy/pandas/scikit-learn as
needed. All input paths are command-line arguments; no analysis path is
hardcoded. Example:

```bash
python ablation/collect_result.py \
    --runs-root <ablation/results> \
    --experiments ablation/experiments.tsv \
    --out ablation/results.csv
```

Each script prints the paths it reads and writes. The original Slurm wrappers
only orchestrated the same Python entry points and are not part of this
package.

