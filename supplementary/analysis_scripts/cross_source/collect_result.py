#!/usr/bin/env python3
"""Summarize the cross-source transfer experiment.

Reads per-seed classification reports from a results directory and writes the
four train/evaluation cells with mean and sample standard deviation over
seeds 40-44.

Usage:
    python collect_result.py --results-root <cross_source/results> --out results.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

ARMS = ("pdb", "uniprot")
HEADER = ["train", "val_eval", "kind", "n", "n_seeds", "seeds",
          "Macro-MCC_mean", "Macro-MCC_sd",
          "Macro-MCC_seed40", "Macro-MCC_seed41", "Macro-MCC_seed42",
          "Macro-MCC_seed43", "Macro-MCC_seed44",
          "Macro-F1_mean", "Macro-F1_sd", "Accuracy_mean", "Accuracy_sd"]


def n_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--seeds", nargs="+", type=int, default=[40, 41, 42, 43, 44])
    args = ap.parse_args()

    rows = []
    for train in ARMS:
        for val in ARMS:
            mcc, f1, acc = [], [], []
            n = None
            for seed in args.seeds:
                base = os.path.join(args.results_root, "seed%d" % seed,
                                    "%s_train" % train, "val_eval_%s" % val)
                rep = os.path.join(base, "classification_report.csv")
                pred = os.path.join(base, "prediction_results.csv")
                if not os.path.exists(rep):
                    continue
                d = common.read_report(rep)
                mcc.append(d["Macro-MCC"])
                f1.append(d["Macro-F1"])
                acc.append(d["Accuracy"])
                if n is None:
                    n = n_rows(pred)
            if not mcc:
                print("warning: no reports for %s->%s" % (train, val))
                continue
            mcc_mean, mcc_sd = common.mean_std(mcc)
            f1_mean, f1_sd = common.mean_std(f1)
            acc_mean, acc_sd = common.mean_std(acc)
            kind = "within-source" if train == val else "cross-source"
            rows.append([
                train, val, kind, n, len(mcc),
                "%d-%d" % (args.seeds[0], args.seeds[-1]),
                "%.4f" % mcc_mean, "%.4f" % mcc_sd,
                *["%.4f" % v for v in mcc],
                "%.4f" % f1_mean, "%.4f" % f1_sd,
                "%.4f" % acc_mean, "%.4f" % acc_sd,
            ])

    common.write_csv(args.out, HEADER, rows)


if __name__ == "__main__":
    main()
