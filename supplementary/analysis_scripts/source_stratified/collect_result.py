#!/usr/bin/env python3
"""Summarize CAPOS performance by annotation source.

Reads the validation predictions and the validation split's Source column,
then writes overall, PDB-derived, and UniProt-derived metrics.

Usage:
    python collect_result.py --pred <prediction_results.csv> --val-split <val.csv> --out results.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common


def read_sources(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["ID"].strip(): (r.get("Source") or "").strip()
                for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--val-split", required=True)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    sources = read_sources(args.val_split)
    pred = common.load_predictions(args.pred)
    tags = [("ALL", np.ones(len(pred["ids"]), dtype=bool))]
    for source in ("PDB", "UniProt"):
        tags.append((source, np.array([sources.get(i) == source
                                       for i in pred["ids"]])))

    rows = []
    for tag, mask in tags:
        if not mask.any():
            continue
        met = common.metrics(pred["y_true"][mask], pred["y_pred"][mask],
                             pred["probs"][mask])
        rows.append(["CAPOS", tag, int(mask.sum()),
                     "%.4f" % met["Macro-MCC"], "%.4f" % met["Macro-F1"],
                     "%.4f" % met["AUC-PR"], "%.4f" % met["Accuracy"]])

    common.write_csv(args.out,
                     ["model", "subset", "n", "Macro-MCC", "Macro-F1",
                      "AUC-PR", "Accuracy"], rows)


if __name__ == "__main__":
    main()
