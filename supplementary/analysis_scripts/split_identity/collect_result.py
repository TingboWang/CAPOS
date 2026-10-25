#!/usr/bin/env python3
"""Summarize nearest-neighbour sequence identity between dataset splits.

Reads the per-query identity CSVs written by compute_identity.py and writes
one summary row per split and split pair.

Usage:
    python collect_result.py --identity-dir <identity> --out results.csv
"""
import argparse
import csv
import os
import statistics

PAIRS = [("train_val", "val->train"), ("train_test", "test->train"),
         ("val_test", "val->test")]
HEADER = ["split", "pair", "n", "no_hit", "ge30_pct", "ge50_pct",
          "ge90_pct", "median_identity"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity-dir", required=True)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    rows = []
    for mode in ("cluster", "random"):
        for stem, label in PAIRS:
            path = os.path.join(args.identity_dir, mode, stem + ".csv")
            vals, no_hit = [], 0
            with open(path, newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    v = (r.get("max_identity") or "").strip()
                    if not v:
                        no_hit += 1
                    else:
                        vals.append(float(v))
            n = len(vals) + no_hit
            rows.append([
                mode, label, n, no_hit,
                "%.1f" % (100.0 * sum(v >= 0.30 for v in vals) / n),
                "%.1f" % (100.0 * sum(v >= 0.50 for v in vals) / n),
                "%.1f" % (100.0 * sum(v >= 0.90 for v in vals) / n),
                "%.3f" % statistics.median(vals),
            ])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print("wrote %s (%d rows)" % (args.out, len(rows)))


if __name__ == "__main__":
    main()
