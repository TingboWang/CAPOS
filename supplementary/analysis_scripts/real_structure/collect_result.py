#!/usr/bin/env python3
"""Summarize the predicted-versus-experimental structure 2x2 experiment.

Reads per-seed prediction files for both training arms and both evaluation
sources, computes the four cells and the evaluation/training effects, and
writes the five-seed mean and sample standard deviation.

Usage:
    python collect_result.py --results-root <real_structure/results> --alignment <alignments_val.csv> --out results.csv
"""
import argparse
import csv
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

STRATA = ["ALL", "pass=True  (Chai-1 confident)",
          "pass=False (Chai-1 not confident)"]
CELLS = [("exp_train", "exp"), ("exp_train", "chai"),
         ("chai_train", "exp"), ("chai_train", "chai")]
SEEDS = [40, 41, 42, 43, 44]


def load_meta(path):
    meta = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r.get("is_selected", "")).strip() not in ("True", "true", "1"):
                continue
            meta[r["ID"].strip()] = str(r.get("pass", "")).strip() == "True"
    return meta


def mask_for(ids, meta, stratum):
    if stratum == "ALL":
        return np.ones(len(ids), dtype=bool)
    want = stratum.startswith("pass=True")
    return np.array([meta.get(pid, True) == want for pid in ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--alignment", required=True)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    meta = load_meta(args.alignment)
    per = {}
    for seed in args.seeds:
        for arm, ev in CELLS:
            path = os.path.join(args.results_root, "seed%d" % seed,
                                arm, "eval_%s" % ev, "prediction_results.csv")
            pred = common.load_predictions(path)
            for stratum in STRATA:
                mask = mask_for(pred["ids"], meta, stratum)
                met = common.metrics(pred["y_true"][mask], pred["y_pred"][mask])
                per[(seed, stratum, arm, ev)] = (met["Macro-MCC"],
                                                 met["Macro-F1"],
                                                 met["Accuracy"], int(mask.sum()))

    rows = []
    for stratum in STRATA:
        for arm, ev in CELLS:
            mcc = [per[(s, stratum, arm, ev)][0] for s in args.seeds]
            f1 = [per[(s, stratum, arm, ev)][1] for s in args.seeds]
            acc = [per[(s, stratum, arm, ev)][2] for s in args.seeds]
            rows.append({
                "kind": "cell", "stratum": stratum, "train": arm, "eval": ev,
                "n": per[(args.seeds[0], stratum, arm, ev)][3],
                "Macro-MCC_mean": "%.4f" % statistics.mean(mcc),
                "Macro-MCC_sd": "%.4f" % statistics.stdev(mcc),
                "Macro-F1_mean": "%.4f" % statistics.mean(f1),
                "Macro-F1_sd": "%.4f" % statistics.stdev(f1),
                "Accuracy_mean": "%.4f" % statistics.mean(acc),
                "Accuracy_sd": "%.4f" % statistics.stdev(acc),
                "seeds": "%d-%d" % (args.seeds[0], args.seeds[-1]),
            })

    def paired(a, b):
        diff = [per[(s, *a)][0] - per[(s, *b)][0] for s in args.seeds]
        return statistics.mean(diff), statistics.stdev(diff), sum(x > 0 for x in diff)

    for stratum in STRATA:
        for name, a, b in [
            ("eval", (stratum, "exp_train", "exp"), (stratum, "exp_train", "chai")),
            ("eval", (stratum, "chai_train", "exp"), (stratum, "chai_train", "chai")),
            ("train", (stratum, "exp_train", "exp"), (stratum, "chai_train", "exp")),
            ("train", (stratum, "exp_train", "chai"), (stratum, "chai_train", "chai")),
        ]:
            mean, sd, wins = paired(a, b)
            rows.append({
                "kind": "effect_%s" % name, "stratum": stratum,
                "train": a[1], "eval": a[2], "n": per[(args.seeds[0], *a)][3],
                "Macro-MCC_mean": "%+.4f" % mean,
                "Macro-MCC_sd": "%.4f" % sd, "wins": wins,
                "seeds": "%d-%d" % (args.seeds[0], args.seeds[-1]),
            })

    fields = ["kind", "stratum", "train", "eval", "n",
              "Macro-MCC_mean", "Macro-MCC_sd", "Macro-F1_mean", "Macro-F1_sd",
              "Accuracy_mean", "Accuracy_sd", "wins", "seeds"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("wrote %s (%d rows)" % (args.out, len(rows)))


if __name__ == "__main__":
    main()
