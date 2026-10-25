#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the ESM-S feature files for the predicted-versus-experimental 2x2.

Writes train_seq.h5 and val_seq.h5 containing embeddings for exactly the
proteins listed in the input CSVs.

Usage:
    python assemble_seq_features.py [--features-dir <dir>] [--esms <h5>] [--out-dir <dir>]
"""

import argparse
import csv
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FEAT = os.path.join(HERE, "features")
ESMS = os.path.join(ROOT, "features", "esms", "raw_esms.h5")


def dec(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default=FEAT)
    ap.add_argument("--esms", default=ESMS)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    feat_dir = os.path.abspath(args.features_dir)
    out_dir = os.path.abspath(args.out_dir or feat_dir)
    os.makedirs(out_dir, exist_ok=True)
    esms = args.esms

    if not os.path.exists(esms):
        sys.exit("missing %s" % esms)
    with h5py.File(esms, "r") as h:
        ids = [dec(x).strip() for x in h["ids"][:]]
        feats = h["features"][:]
        labels = h["labels"][:]
    idx = {pid: i for i, pid in enumerate(ids)}
    print("ESM-S source     : %s  (%d x %d)" % (os.path.relpath(esms, ROOT),
                                               len(ids), feats.shape[1]))

    for sp in ("train", "val"):
        src = os.path.join(feat_dir, "input_%s.csv" % sp)
        rows = list(csv.DictReader(open(src, newline="", encoding="utf-8-sig")))
        want = [(r["ID"].strip(), int(r["Label"])) for r in rows]

        keep_i, keep_id, keep_lab = [], [], []
        missing = bad_label = 0
        for pid, lab in want:
            i = idx.get(pid)
            if i is None:
                missing += 1
                continue
            if int(labels[i]) != lab:
                bad_label += 1
                continue
            keep_i.append(i); keep_id.append(pid); keep_lab.append(lab)

        if missing:
            sys.exit("!! %d IDs of %s are absent from %s -- refusing to continue"
                     % (missing, sp, os.path.relpath(esms, ROOT)))
        if bad_label:
            sys.exit("!! %d rows of %s disagree with %s labels -- refusing to continue"
                     % (bad_label, sp, os.path.relpath(esms, ROOT)))

        out = os.path.join(out_dir, "%s_seq.h5" % sp)
        with h5py.File(out, "w") as h:
            h.create_dataset("ids", data=np.array(keep_id, dtype=object),
                             dtype=h5py.string_dtype(encoding="utf-8"))
            h.create_dataset("labels", data=np.array(keep_lab, dtype=np.int64))
            h.create_dataset("features", data=feats[np.array(keep_i, dtype=np.int64)],
                             compression="gzip", compression_opts=4)
        print("%-6s -> %-34s  %d x %d  (kept %d of %d, missing %d)"
              % (sp, os.path.relpath(out, ROOT), len(keep_id), feats.shape[1],
                 len(keep_id), len(want), missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
