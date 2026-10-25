#!/usr/bin/env python3
"""Summarize the CAPOS ablation runs into one results table.

Reads per-seed classification reports from a runs directory and writes mean
and sample standard deviation over seeds 40-44. Validated on the validation
split only.

Usage:
    python collect_result.py --runs-root <ablation/results> --experiments experiments.tsv --out results.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

METRICS = ["MCC", "F1-score", "AUC-PR", "Accuracy", "Precision", "Recall"]
OUT_NAMES = {"MCC": "Macro_MCC", "F1-score": "Macro_F1", "AUC-PR": "AUC_PR"}
HEADER = ["group", "run"]
for _m in METRICS:
    HEADER += [OUT_NAMES.get(_m, _m) + "_mean", OUT_NAMES.get(_m, _m) + "_std"]

GROUP_ORDER = {"reference": 0, "fusion_operator": 1,
               "channel_control": 2, "sequence_encoder": 3}
RUN_ORDER = [
    "esms_saprot_normal",
    "esms_saprot_normal_fixed_gate", "esms_saprot_normal_concat",
    "esms_saprot_mask3di", "esms_saprot_shuffle3di", "esms_saprot_maskaa",
    "esms_only", "saprot_normal_only", "saprot_mask3di_only",
    "saprot_maskaa_only", "saprot_shuffle3di_only",
    "prott5_saprot_normal", "esm2_saprot_normal", "esmc_saprot_normal",
]


def read_experiments(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def group_for(name, mode):
    if name == "esms_saprot_normal":
        return "reference"
    if mode in ("fixed_gate", "concat"):
        return "fusion_operator"
    if name in ("esm2_saprot_normal", "esmc_saprot_normal",
                "prott5_saprot_normal"):
        return "sequence_encoder"
    return "channel_control"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--experiments", required=True)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    args = ap.parse_args()

    rows = []
    for exp in read_experiments(args.experiments):
        run, mode = exp["name"], exp["ablation_mode"]
        per_metric = {m: [] for m in METRICS}
        for seed in args.seeds:
            path = os.path.join(args.runs_root, run, "seed%d" % seed,
                                "classification_report.csv")
            if not os.path.exists(path):
                continue
            rep = common.read_report(path)
            for m in METRICS:
                per_metric[m].append(rep[m])
        if not per_metric["MCC"]:
            print("warning: no reports for %s" % run)
            continue
        row = [group_for(run, mode), run]
        for m in METRICS:
            mean, std = common.mean_std(per_metric[m])
            row += ["%.4f" % mean, "%.4f" % std]
        rows.append(row)

    rows.sort(key=lambda r: (GROUP_ORDER.get(r[0], 9),
                             RUN_ORDER.index(r[1]) if r[1] in RUN_ORDER else 99))
    common.write_csv(args.out, HEADER, rows)


if __name__ == "__main__":
    main()
