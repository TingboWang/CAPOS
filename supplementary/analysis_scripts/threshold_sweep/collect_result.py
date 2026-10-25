#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate five-seed validation results for the dG threshold sweep.

Reads <sweep-dir>/<threshold>/seed<S>/classification_report.csv and writes a
single CSV with mean and sample standard deviation over seeds 40-44.

Usage:
    python collect_result.py --sweep-dir <dir> --out results.csv
"""
import argparse
import csv
import os
import statistics

THRESHOLDS = ["raw", "0", "5", "10", "20", "30"]
SEEDS = [40, 41, 42, 43, 44]
SELECTED = "5"


def read_metrics(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = {r[""]: r for r in csv.DictReader(f)}
    macro = rows.get("Macro avg", {})
    accuracy = rows.get("Accuracy", {})
    return {
        "Macro_MCC": float(macro.get("MCC") or 0.0),
        "Macro_F1": float(macro.get("F1-score") or 0.0),
        "Accuracy": float(accuracy.get("Precision") or 0.0),
        "AUC_PR": float(macro.get("AUC-PR") or 0.0),
    }


def mean_std(values):
    if not values:
        return "", ""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else ""
    return "%.4f" % mean, ("%.4f" % std if std != "" else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True,
                    help="directory containing <threshold>/seed<S>/classification_report.csv")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    header = ["threshold", "n_seeds",
              "Macro_MCC_mean", "Macro_MCC_std",
              "Macro_F1_mean", "Macro_F1_std",
              "Accuracy_mean", "Accuracy_std",
              "AUC_PR_mean", "AUC_PR_std", "selected"]
    rows = []
    for threshold in THRESHOLDS:
        values = {k: [] for k in ("Macro_MCC", "Macro_F1", "Accuracy", "AUC_PR")}
        for seed in SEEDS:
            path = os.path.join(args.sweep_dir, threshold, "seed%d" % seed,
                                "classification_report.csv")
            if not os.path.exists(path):
                continue
            metrics = read_metrics(path)
            for key in values:
                values[key].append(metrics[key])
        if not values["Macro_MCC"]:
            print("warning: no reports found for threshold %s" % threshold)
            continue
        row = [threshold, len(values["Macro_MCC"])]
        for key in ("Macro_MCC", "Macro_F1", "Accuracy", "AUC_PR"):
            row += list(mean_std(values[key]))
        row.append("yes" if threshold == SELECTED else "no")
        rows.append(row)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print("wrote %s (%d thresholds)" % (args.out, len(rows)))


if __name__ == "__main__":
    main()
