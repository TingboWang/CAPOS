#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the held-out evaluation sets for the cross-source experiment.

Writes cluster-pure PDB and UniProt validation subsets that are disjoint from
the corresponding training clusters.

Usage:
    python make_val_subsets.py [--base <CAPOS root>] [--out-dir <dir>]
"""

import argparse
import csv
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
VAL_CSV = None
SELECTION_CSV = None
ESMS = None
SAPROT = None
OUT_DIR = HERE

CLASSES = [1, 2, 3, 4, 5, 6, 8, 10, 12]


def dec(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    global ROOT, VAL_CSV, SELECTION_CSV, ESMS, SAPROT, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=ROOT, help="CAPOS repository root")
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()
    ROOT = os.path.abspath(args.base)
    OUT_DIR = os.path.abspath(args.out_dir)
    VAL_CSV = os.path.join(ROOT, "dataset", "dataset_split", "splits", "val.csv")
    SELECTION_CSV = os.path.join(ROOT, "monomer_structures",
                                 "structure_selection.csv")
    ESMS = os.path.join(ROOT, "features", "esms", "raw_esms.h5")
    SAPROT = os.path.join(ROOT, "features", "saprot", "raw_saprot.h5")

    os.makedirs(os.path.join(OUT_DIR, "splits"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "features"), exist_ok=True)

    rows = read_csv(VAL_CSV)
    print("validation pool    : %d rows" % len(rows))

    pass_of = {}
    for r in read_csv(SELECTION_CSV):
        pass_of[r["ID"].strip()] = (r.get("pass") or "").strip() == "True"

    kept, dropped_no_pass = [], 0
    for r in rows:
        pid = r["ID"].strip()
        if not pass_of.get(pid, False):
            dropped_no_pass += 1
            continue
        kept.append({
            "ID": pid,
            "Sequence": r["Sequence"].strip(),
            "Label": int(r["Label"]),
            "Source": r["Source"].strip(),
            "Cluster_Num": r["Cluster_Num"].strip(),
        })
    print("  after pass filter: %d rows (dropped %d)"
          % (len(kept), dropped_no_pass))

    # ---- cluster purity, same rule as the training subsets
    by_cluster = {}
    for r in kept:
        by_cluster.setdefault(r["Cluster_Num"], []).append(r)

    subsets = {"PDB": [], "UniProt": []}
    mixed = []
    for members in by_cluster.values():
        srcs = {m["Source"] for m in members}
        if len(srcs) == 1:
            subsets[members[0]["Source"]].extend(members)
        else:
            mixed.extend(members)

    print("\ncluster purity (validation, after pass filter):")
    print("  clusters total   : %d" % len(by_cluster))
    print("  pure PDB         : %d clusters -> %d sequences"
          % (sum(1 for m in by_cluster.values()
                 if len({x['Source'] for x in m}) == 1
                 and m[0]['Source'] == 'PDB'), len(subsets["PDB"])))
    print("  pure UniProt     : %d clusters -> %d sequences"
          % (sum(1 for m in by_cluster.values()
                 if len({x['Source'] for x in m}) == 1
                 and m[0]['Source'] == 'UniProt'), len(subsets["UniProt"])))
    print("  mixed (dropped)  : %d clusters -> %d sequences"
          % (sum(1 for m in by_cluster.values()
                 if len({x['Source'] for x in m}) > 1), len(mixed)))

    # ---- I1/I2: nothing in validation may reach back into training
    train_ids, train_cls = set(), set()
    for name in ("pdb", "uniprot"):
        p = os.path.join(OUT_DIR, "splits", "%s.csv" % name)
        if not os.path.exists(p):
            sys.exit("!! missing %s -- run make_subsets.py first" % p)
        for r in read_csv(p):
            train_ids.add(r["ID"].strip())
            train_cls.add(r["Cluster_Num"].strip())

    val_ids = [r["ID"] for r in kept]
    val_cls = [r["Cluster_Num"] for r in kept]
    hit_id = set(val_ids) & train_ids
    hit_cl = set(val_cls) & train_cls
    if hit_id:
        sys.exit("!! I2 violated: %d validation ids are in the trained subsets"
                 % len(hit_id))
    if hit_cl:
        sys.exit("!! I1 violated: %d validation clusters are in the trained "
                 "subsets" % len(hit_cl))
    print("\ninvariants I1-I2 ok (validation is disjoint from both trained "
          "subsets)")

    # ---- I3-I4
    pdb_ids = [r["ID"] for r in subsets["PDB"]]
    uni_ids = [r["ID"] for r in subsets["UniProt"]]
    if set(pdb_ids) & set(uni_ids):
        sys.exit("!! I3 violated: an ID is in both val subsets")
    if {r["Cluster_Num"] for r in subsets["PDB"]} & \
       {r["Cluster_Num"] for r in subsets["UniProt"]}:
        sys.exit("!! I3 violated: a cluster is in both val subsets")
    if any(r["Source"] != "PDB" for r in subsets["PDB"]):
        sys.exit("!! I4 violated: non-PDB row in the PDB subset")
    if any(r["Source"] != "UniProt" for r in subsets["UniProt"]):
        sys.exit("!! I4 violated: non-UniProt row in the UniProt subset")
    print("invariants I3-I4 ok")

    def write_csv(path, rows_, cols):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows_:
                w.writerow({c: r[c] for c in cols})

    cols = ["ID", "Sequence", "Label", "Source", "Cluster_Num"]
    write_csv(os.path.join(OUT_DIR, "splits", "val_pdb.csv"), subsets["PDB"], cols)
    write_csv(os.path.join(OUT_DIR, "splits", "val_uniprot.csv"),
              subsets["UniProt"], cols)
    write_csv(os.path.join(OUT_DIR, "splits", "val_dropped_mixed.csv"), mixed, cols)
    print("wrote splits/val_{pdb,uniprot,dropped_mixed}.csv")

    # ---- I5: report class coverage honestly, thin classes are not hidden
    print()
    for name, rows_ in (("PDB", subsets["PDB"]), ("UniProt", subsets["UniProt"])):
        counts = {c: sum(1 for r in rows_ if r["Label"] == c) for c in CLASSES}
        thin = [c for c in CLASSES if 0 < counts[c] < 30]
        zero = [c for c in CLASSES if counts[c] == 0]
        print("%-7s class counts: %s"
              % (name, "  ".join("%d:%d" % (c, counts[c]) for c in CLASSES)))
        if zero:
            print("        !! I5 note: class %s has no rows at all -- its "
                  "per-class score will be degenerate" % zero)
        if thin:
            print("        !! I5 note: class %s has fewer than 30 rows"
                  % thin)
    print("invariants I5-I7 checked below")

    # ---- features
    def load_raw(path, tag):
        if not os.path.exists(path):
            sys.exit("missing %s" % path)
        with h5py.File(path, "r") as h:
            ids = [dec(x).strip() for x in h["ids"][:]]
            feats = h["features"][:]
            labs = h["labels"][:]
        print("\n%s source: %s  (%d x %d)"
              % (tag, os.path.relpath(path, ROOT), len(ids), feats.shape[1]))
        return {p: i for i, p in enumerate(ids)}, feats, labs

    esm_idx, esm_feats, esm_labs = load_raw(ESMS, "ESM-S")
    sap_idx, sap_feats, sap_labs = load_raw(SAPROT, "SaProt")

    for name, key in (("PDB", "pdb"), ("UniProt", "uniprot")):
        rows_ = subsets[name]
        want = [(r["ID"], r["Label"]) for r in rows_]
        out_i, out_id, out_lab = [], [], []
        missing = bad = 0
        for pid, lab in want:
            ie, is_ = esm_idx.get(pid), sap_idx.get(pid)
            if ie is None or is_ is None:
                missing += 1
                continue
            if int(esm_labs[ie]) != lab or int(sap_labs[is_]) != lab:
                bad += 1
                continue
            out_i.append((ie, is_))
            out_id.append(pid)
            out_lab.append(lab)
        if missing:
            sys.exit("!! I7 violated: %d %s ids absent from the raw feature "
                     "files" % (missing, name))
        if bad:
            sys.exit("!! I6 violated: %d %s rows disagree with the raw "
                     "feature labels" % (bad, name))

        ei = np.array([p[0] for p in out_i], dtype=np.int64)
        si = np.array([p[1] for p in out_i], dtype=np.int64)
        for branch, src_feats, sel in (("seq", esm_feats, ei),
                                       ("struct", sap_feats, si)):
            out = os.path.join(OUT_DIR, "features",
                               "val_%s_%s.h5" % (key, branch))
            with h5py.File(out, "w") as h:
                h.create_dataset("ids", data=np.array(out_id, dtype=object),
                                 dtype=h5py.string_dtype(encoding="utf-8"))
                h.create_dataset("labels", data=np.array(out_lab, dtype=np.int64))
                h.create_dataset("features", data=src_feats[sel],
                                 compression="gzip", compression_opts=4)
            print("  %-12s -> %-42s %d x %d"
                  % (branch, os.path.relpath(out, ROOT),
                     len(out_id), src_feats.shape[1]))
    print("\ninvariants I6-I7 ok")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
