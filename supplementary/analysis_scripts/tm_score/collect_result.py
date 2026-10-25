#!/usr/bin/env python3
"""Summarize CAPOS performance across experimental TM-score bins.

Reads per-protein TM-scores and the matching CAPOS predictions, bins by
TM-score, and writes accuracy, Macro MCC, Macro F1, and the median TM-score.

Usage:
    python collect_result.py --tm-csv <tm_scores_val.csv> --pred <prediction_results.csv> --out results.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

BINS = [("<0.50", -np.inf, 0.50), ("0.50-0.60", 0.50, 0.60),
        ("0.60-0.70", 0.60, 0.70), ("0.70-0.80", 0.70, 0.80),
        ("0.80-0.90", 0.80, 0.90), ("0.90-0.95", 0.90, 0.95),
        (">=0.95", 0.95, np.inf)]


def fmt(value):
    return ("%.4f" % value).rstrip("0").rstrip(".")


def read_tm(path):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if (r.get("status") or "").strip() not in ("ok", ""):
                continue
            value = (r.get("tm_score") or "").strip()
            if value:
                out[r["ID"].strip()] = float(value)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm-csv", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    tm = read_tm(args.tm_csv)
    pred = common.load_predictions(args.pred)
    index = {pid: i for i, pid in enumerate(pred["ids"])}
    pairs = [(index[pid], tm[pid]) for pid in tm if pid in index]

    rows = []
    for order, (label, lo, hi) in enumerate(BINS, 1):
        sel = [(idx, value) for idx, value in pairs if lo <= value < hi]
        if not sel:
            continue
        idxs = [idx for idx, _ in sel]
        yt = pred["y_true"][idxs]
        yp = pred["y_pred"][idxs]
        met = common.metrics(yt, yp)
        rows.append([label, len(sel), fmt(np.median([v for _, v in sel])),
                     fmt(met["Accuracy"]), fmt(met["Macro-MCC"]),
                     fmt(met["Macro-F1"]), order])

    common.write_csv(args.out,
                     ["TM bin", "n", "tm_median", "Accuracy", "Macro-MCC",
                      "Macro-F1", "bin_order"], rows)


if __name__ == "__main__":
    main()
