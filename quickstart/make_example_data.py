#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the ten-protein quickstart inputs out of the repository's feature store.

    python make_example_data.py --n 10

Cut N example proteins out of a feature store and write them in the schema
the CLI reads, so `capos predict` sees exactly the file layout documented in
the README. Kept for provenance: the quickstart inputs ship with the release, so
this is only needed to rebuild them.

This quickstart is a smoke test of the plumbing, not a benchmark.

Inputs
------
    --esms       ESM-S feature store (.h5)
    --saprot     SaProt feature store (.h5)
    --sequences  CSV of ID, Sequence
    --pred       CSV of reference predictions, for the truth table

Output, into --out-dir
----------------------
    example_esms.h5        N x 1280
    example_saprot.h5      N x 1280
    example.fasta          the same N sequences
    example_truth.csv      the reference predictions, for eyeballing
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def dec(x):
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=10,
                    help="how many proteins to cut out (default: 10)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "example_data"),
                    help="where to write the quickstart files")
    ap.add_argument("--seed", type=int, default=42,
                    help="which rows to pick (default: 42)")
    ap.add_argument("--esms",
                    default=os.path.join(REPO, "features", "esms",
                                         "raw_esms.h5"),
                    help="ESM-S feature store (.h5)")
    ap.add_argument("--saprot",
                    default=os.path.join(REPO, "features", "saprot",
                                         "raw_saprot.h5"),
                    help="SaProt feature store (.h5)")
    ap.add_argument("--sequences",
                    default=os.path.join(REPO, "dataset", "dataset_split",
                                         "splits", "test.csv"),
                    help="CSV with ID, Sequence columns")
    ap.add_argument("--pred",
                    default=os.path.join(REPO, "model", "results", "sweep",
                                         "5", "seed42", "test",
                                         "prediction_results.csv"),
                    help="CSV of reference predictions, for the truth table")
    a = ap.parse_args()

    import h5py

    esms, saprot = a.esms, a.saprot
    truth, split = a.pred, a.sequences
    missing = [p for p in (esms, saprot, truth, split)
               if not os.path.exists(p)]
    if missing:
        for p in missing:
            print("!! missing %s" % p)
        sys.exit("   pass --esms / --saprot / --sequences / --pred, or run "
                 "this from a checkout that has those files")

    # prediction_results.csv has no sequence column; the sequences come from
    # the sequence table.
    seq_of = {r["ID"].strip(): r["Sequence"].strip()
              for r in csv.DictReader(open(split, encoding="utf-8-sig"))}

    rows = list(csv.DictReader(open(truth, encoding="utf-8-sig")))
    rng = np.random.default_rng(a.seed)
    pick = [rows[i] for i in rng.choice(len(rows), size=a.n, replace=False)]
    want = [r["ID"].strip() for r in pick]
    print("picked %d ids: %s" % (len(want), ", ".join(want[:6]) + ("..." if a.n > 6 else "")))

    os.makedirs(a.out_dir, exist_ok=True)

    def cut(src, out):
        with h5py.File(src, "r") as h:
            ids = [dec(x).strip() for x in h["ids"][:]]
            feats = h["features"][:]
            labs = h["labels"][:]
        pos = {i: k for k, i in enumerate(ids)}
        missing = [i for i in want if i not in pos]
        if missing:
            sys.exit("!! %s lacks %s" % (src, missing))
        sel = [pos[i] for i in want]
        with h5py.File(out, "w") as h:
            h.create_dataset("ids", data=np.array(want, dtype=object),
                             dtype=h5py.string_dtype(encoding="utf-8"))
            h.create_dataset("labels", data=labs[sel].astype(np.int64))
            h.create_dataset("features", data=feats[sel].astype(np.float32),
                             compression="gzip", compression_opts=4)
        print("  %-16s -> %s  (%d x %d)"
              % (os.path.basename(src), os.path.relpath(out, REPO),
                 len(want), feats.shape[1]))

    cut(esms, os.path.join(a.out_dir, "example_esms.h5"))
    cut(saprot, os.path.join(a.out_dir, "example_saprot.h5"))

    with open(os.path.join(a.out_dir, "example.fasta"), "w") as f:
        for r in pick:
            pid = r["ID"].strip()
            seq = seq_of.get(pid)
            if not seq:
                sys.exit("!! %s has no sequence in the test split" % pid)
            f.write(">%s\n%s\n" % (pid, seq))

    with open(os.path.join(a.out_dir, "example_truth.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "True_Label", "Reference_Prediction"])
        for r in pick:
            w.writerow([r["ID"].strip(), r["True_Label"], r["Pred_Label"]])

    print("\nwrote quickstart inputs to %s" % a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
