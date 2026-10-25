#!/usr/bin/env python3
"""Summarize CAPOS and Foldseek-Trans performance by structural similarity.

Uses the best Foldseek hit for each test protein, ranks by
tm_mean=(qTM+tTM)/2, and reports both methods inside each similarity stratum.

Usage:
    python collect_result.py --best-hit-tmscore <best_hit_tmscore.csv> --capos-pred <prediction_results.csv> --foldseek-pred <prediction_results.csv> --out results.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

STRATA = [
    ("no_hit", None, 0.0),
    ("tm_lt_0.50", 0.0, 0.50),
    ("tm_0.50_0.70", 0.50, 0.70),
    ("tm_0.70_0.90", 0.70, 0.90),
    ("tm_ge_0.90", 0.90, 10.0),
    ("no_or_weak_homology", 0.0, 0.50),
    ("strong_homology", 0.90, 10.0),
]


def read_tm(path):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            value = (r.get("tm_mean") or "").strip()
            out[r["ID"].strip()] = float(value) if value else None
    return out


def scores(pred, index):
    yt = pred["y_true"][index]
    yp = pred["y_pred"][index]
    met = common.metrics(yt, yp)
    return met["Macro-MCC"], met["Macro-F1"], met["Accuracy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-hit-tmscore", required=True)
    ap.add_argument("--capos-pred", required=True)
    ap.add_argument("--foldseek-pred", required=True)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    tm = read_tm(args.best_hit_tmscore)
    capos = common.load_predictions(args.capos_pred)
    fold = common.load_predictions(args.foldseek_pred)
    capos_index = {pid: i for i, pid in enumerate(capos["ids"])}
    fold_index = {pid: i for i, pid in enumerate(fold["ids"])}
    ids = [pid for pid in capos["ids"] if pid in fold_index]

    rows = []
    for name, lo, hi in STRATA:
        if name == "no_hit":
            sel = [pid for pid in ids if tm.get(pid) is None]
        elif name == "no_or_weak_homology":
            sel = [pid for pid in ids
                   if tm.get(pid) is None or tm[pid] < 0.50]
        else:
            sel = [pid for pid in ids
                   if tm.get(pid) is not None and lo <= tm[pid] < hi]
        if not sel:
            continue
        ci = [capos_index[pid] for pid in sel]
        fi = [fold_index[pid] for pid in sel]
        cm, cf, ca = scores(capos, ci)
        fm, ff, fa = scores(fold, fi)
        rows.append([name, "" if lo is None else str(lo),
                     str(hi), len(sel),
                     str(cm), str(fm), str(cm - fm),
                     str(cf), str(ff), str(cf - ff),
                     str(ca), str(fa), str(ca - fa)])

    common.write_csv(args.out,
                     ["stratum", "tm_lo", "tm_hi", "n",
                      "capos_macro_mcc", "foldseek_macro_mcc", "delta_macro_mcc",
                      "capos_macro_f1", "foldseek_macro_f1", "delta_macro_f1",
                      "capos_accuracy", "foldseek_accuracy", "delta_accuracy"],
                     rows)


if __name__ == "__main__":
    main()
