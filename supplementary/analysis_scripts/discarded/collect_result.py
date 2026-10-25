#!/usr/bin/env python3
"""Summarize CAPOS performance on proteins excluded by the structure-quality gate.

Reads the excluded-set predictions, keeps only sequences with no trained-on
cluster mate ("unseen"), and writes the kept-versus-excluded comparison.

Usage:
    python collect_result.py --excluded-pred <prediction_results.csv> --profile <discarded_profile.csv> --kept-report <classification_report.csv> --out results.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common


def exposed_ids(path):
    out = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if (r.get("cluster_has_trained_on") or "").strip().lower() in ("yes", "true", "1"):
                out.add(r["ID"].strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excluded-pred", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--kept-report", required=True)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    pred = common.load_predictions(args.excluded_pred)
    exposed = exposed_ids(args.profile)
    keep = [i for i, pid in enumerate(pred["ids"]) if pid not in exposed]
    yt, yp = pred["y_true"][keep], pred["y_pred"][keep]
    pr = pred["probs"][keep]
    met = common.metrics(yt, yp, pr)

    kept = common.read_report(args.kept_report)
    rows = [
        ["kept_validation", kept["n"],
         "%.4f" % kept["Macro-MCC"], "%.4f" % kept["Macro-F1"],
         "%.4f" % kept["Accuracy"], "%.4f" % kept["AUC-PR"]],
        ["excluded_unseen", len(keep),
         "%.4f" % met["Macro-MCC"], "%.4f" % met["Macro-F1"],
         "%.4f" % met["Accuracy"], "%.4f" % met["AUC-PR"]],
    ]
    common.write_csv(args.out,
                     ["subset", "n", "Macro_MCC", "Macro_F1", "Accuracy", "AUC_PR"],
                     rows)


if __name__ == "__main__":
    main()
