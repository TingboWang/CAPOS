#!/usr/bin/env python3
"""Build row-normalised confusion counts for all benchmark methods.

The input manifest is a CSV with two columns:
    method,prediction_results
Each prediction file must contain True_Label and Pred_Label. The output is a
long-format table with raw counts, per-class support, and row fractions.

Usage:
    python collect_result.py --manifest methods.csv --out results.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

PRED_COLS = common.CLASSES + ["off_label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    with open(args.manifest, newline="", encoding="utf-8-sig") as f:
        methods = list(csv.DictReader(f))

    rows = []
    for entry in methods:
        method = entry["method"].strip()
        pred = common.load_predictions(entry["prediction_results"].strip())
        true = pred["true_raw"]
        pred_label = pred["pred_raw"]
        for t in common.CLASSES:
            idx = [i for i, v in enumerate(true) if v == t]
            n_true = len(idx)
            for p in PRED_COLS:
                if p == "off_label":
                    count = sum(1 for i in idx if pred_label[i] not in common.CLASSES)
                else:
                    count = sum(1 for i in idx if pred_label[i] == int(p))
                frac = (count / n_true) if n_true else 0.0
                rows.append([method, t, p, count, n_true, "%.6f" % frac])

    common.write_csv(args.out,
                     ["method", "true_label", "predicted_label", "count",
                      "n_true", "row_fraction"], rows)


if __name__ == "__main__":
    main()
