#!/usr/bin/env python3
"""Summarize CAPOS structure-quality analyses into one results table.

Sections:
  threshold_grid  validation metrics under pLDDT/pTM cutoffs (seed 42)
  quality_bins    validation metrics in pLDDT and pTM bins (seed 42)
  branch_gain     full / SaProt-only / ESM-S-only in quality bins (seeds 40-44)
  train_quality   validation metrics for training-set pLDDT cutoffs (seeds 40-44)

Usage:
    python collect_result.py --val-pred <prediction_results.csv> --selection <structure_selection.csv> --full-root <sweep/5> --esms-root <ablation/results/esms_only> --saprot-root <ablation/results/saprot_normal_only> --train-quality-root <train_quality> --out results.csv
"""
import argparse
import csv
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

PLDDT_GRID = [70, 75, 80, 85, 90, 95]
PTM_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
PLDDT_BINS = [(70, 80, "70-80"), (80, 85, "80-85"), (85, 90, "85-90"),
              (90, 95, "90-95"), (95, 101, "95-100")]
PTM_BINS = [(0.5, 0.6, "0.5-0.6"), (0.6, 0.7, "0.6-0.7"),
            (0.7, 0.8, "0.7-0.8"), (0.8, 0.9, "0.8-0.9"),
            (0.9, 1.01, "0.9-1.0")]
PTM_BINS_BRANCH = [(lo, hi, "0.9-1.01" if label == "0.9-1.0" else label)
                   for lo, hi, label in PTM_BINS]
PLDDT_BINS_BRANCH = [(lo, hi, "95-101" if label == "95-100" else label)
                     for lo, hi, label in PLDDT_BINS]
SEEDS = [40, 41, 42, 43, 44]
HEADER = ["section", "axis", "bin_or_cut", "n", "kept_pct",
          "Macro_MCC", "Macro_MCC_sd", "Macro_F1", "Macro_F1_sd",
          "Accuracy", "Accuracy_sd", "AUC_PR", "AUC_PR_sd",
          "full_MCC_mean", "full_MCC_sd", "saprot_only_MCC_mean",
          "saprot_only_MCC_sd", "esms_only_MCC_mean", "esms_only_MCC_sd",
          "delta_full_minus_esms", "delta_full_minus_saprot"]


def read_selection(path):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                out[r["ID"].strip()] = (float(r["best_ptm"]),
                                        float(r["best_plddt"]))
            except (TypeError, ValueError):
                continue
    return out


def load(path):
    return common.load_predictions(path)


def subset(pred, keep):
    idx = [i for i, pid in enumerate(pred["ids"]) if keep.get(pid)]
    return pred["y_true"][idx], pred["y_pred"][idx], pred["probs"][idx]


def metrics4(pred, keep):
    yt, yp, pr = subset(pred, keep)
    return common.metrics(yt, yp, pr), len(yt)


def write_section(rows, section, axis, label, n, kept_pct, met,
                  branch=None, full_precision=False):
    def value(x):
        return str(float(x)) if full_precision else "%.4f" % x
    row = [section, axis, label, n, kept_pct,
           value(met["Macro-MCC"]) if met else "",
           "", value(met["Macro-F1"]) if met else "", "",
           value(met["Accuracy"]) if met else "", "",
           value(met["AUC-PR"]) if met else "", "",
           "", "", "", "", "", "", "", ""]
    if branch:
        full_mean, full_sd, sap_mean, sap_sd, esm_mean, esm_sd, de, ds = branch
        row[13:21] = [str(full_mean), str(full_sd), str(sap_mean), str(sap_sd),
                      str(esm_mean), str(esm_sd), str(de), str(ds)]
    rows.append(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-pred", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--full-root", required=True,
                    help="directory with seed<S>/prediction_results.csv")
    ap.add_argument("--esms-root", required=True,
                    help="ESM-S-only run root with seed<S>/prediction_results.csv")
    ap.add_argument("--saprot-root", required=True,
                    help="SaProt-only run root with seed<S>/prediction_results.csv")
    ap.add_argument("--train-quality-root", required=True)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    sel = read_selection(args.selection)
    val = load(args.val_pred)
    n_all = len(val["ids"])

    rows = []
    # threshold grid
    for pld in PLDDT_GRID:
        for ptm in PTM_GRID:
            keep = {pid: (sel.get(pid, (-1, -1))[0] >= ptm and
                          sel.get(pid, (-1, -1))[1] >= pld)
                    for pid in val["ids"]}
            met, n = metrics4(val, keep)
            label = "pLDDT_min=%d; pTM_min=%s" % (pld, ptm)
            write_section(rows, "threshold_grid", label, "", n,
                          str(100.0 * n / n_all), met)

    # quality bins, seed 42
    for axis, bins, getter in (
            ("pLDDT", PLDDT_BINS, lambda v: v[1]),
            ("pTM", PTM_BINS, lambda v: v[0])):
        for lo, hi, label in bins:
            keep = {pid: (pid in sel and lo <= getter(sel[pid]) < hi)
                    for pid in val["ids"]}
            met, n = metrics4(val, keep)
            write_section(rows, "quality_bins", axis, label, n, "", met,
                          full_precision=True)

    # branch gain, 5 seeds
    full = {s: load(os.path.join(args.full_root, "seed%d" % s,
                                 "prediction_results.csv")) for s in SEEDS}
    esms = {s: load(os.path.join(args.esms_root, "seed%d" % s,
                                 "prediction_results.csv")) for s in SEEDS}
    sap = {s: load(os.path.join(args.saprot_root, "seed%d" % s,
                                "prediction_results.csv")) for s in SEEDS}
    for axis, bins, getter in (
            ("pTM", PTM_BINS_BRANCH, lambda v: v[0]),
            ("pLDDT", PLDDT_BINS_BRANCH, lambda v: v[1])):
        for lo, hi, label in bins:
            keep = {pid: (pid in sel and lo <= getter(sel[pid]) < hi)
                    for pid in val["ids"]}
            vals = {"full": [], "esms": [], "saprot": []}
            n = 0
            for s in SEEDS:
                for name, pred in (("full", full[s]), ("esms", esms[s]),
                                   ("saprot", sap[s])):
                    yt, yp, _ = subset(pred, keep)
                    vals[name].append(common.macro_mcc(yt, yp))
                n = sum(1 for pid in full[s]["ids"] if keep.get(pid))
            full_mean, full_sd = common.mean_std(vals["full"])
            sap_mean, sap_sd = common.mean_std(vals["saprot"])
            esm_mean, esm_sd = common.mean_std(vals["esms"])
            write_section(rows, "branch_gain", axis, label, n, "", None,
                          (full_mean, full_sd, sap_mean, sap_sd,
                           esm_mean, esm_sd,
                           full_mean - esm_mean, full_mean - sap_mean))

    # train-set quality control
    for cut in ("70", "80", "90"):
        per = {k: [] for k in ("Macro-MCC", "Macro-F1", "Accuracy", "AUC-PR")}
        for s in SEEDS:
            rep = common.read_report(os.path.join(
                args.train_quality_root, "plddt%s" % cut, "seed%d" % s,
                "classification_report.csv"))
            for k in per:
                per[k].append(rep[k])
        train_csv = os.path.join(args.train_quality_root,
                                 "train_ptm0.5_plddt%s.csv" % cut)
        with open(train_csv, encoding="utf-8-sig") as f:
            n_train = max(0, sum(1 for _ in f) - 1)
        row = ["train_quality", "train_pLDDT_cut", ">= %s" % cut, n_train, ""]
        for k in ("Macro-MCC", "Macro-F1", "Accuracy", "AUC-PR"):
            mean, sd = common.mean_std(per[k])
            row += ["%.4f" % mean, "%.4f" % sd]
        row += [""] * 8
        rows.append(row)

    common.write_csv(args.out, HEADER, rows)


if __name__ == "__main__":
    main()
