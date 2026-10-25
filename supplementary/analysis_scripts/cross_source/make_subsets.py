#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build cluster-pure PDB and UniProt training subsets for cross-source tests.

Starting from the 5 kcal/mol training pool, removes mixed-source clusters so
that the two single-source training sets share no MMseqs2 cluster.

Usage:
    python make_subsets.py [--base <CAPOS root>] [--out-dir <dir>]
"""

import argparse
import csv
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TRAIN_CSV = None
SELECTION_CSV = None
ESMS = None
SAPROT = None
OUT_DIR = HERE

CLASSES = [1, 2, 3, 4, 5, 6, 8, 10, 12]
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 10: 7, 12: 8}


def dec(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    global ROOT, TRAIN_CSV, SELECTION_CSV, ESMS, SAPROT, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=ROOT, help="CAPOS repository root")
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()
    ROOT = os.path.abspath(args.base)
    OUT_DIR = os.path.abspath(args.out_dir)
    TRAIN_CSV = os.path.join(ROOT, "dataset", "dataset_split", "splits",
                             "dataset_5_train.csv")
    SELECTION_CSV = os.path.join(ROOT, "monomer_structures",
                                 "structure_selection.csv")
    ESMS = os.path.join(ROOT, "features", "esms", "raw_esms.h5")
    SAPROT = os.path.join(ROOT, "features", "saprot", "raw_saprot.h5")

    os.makedirs(os.path.join(OUT_DIR, "splits"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "features"), exist_ok=True)

    # ---- 1) the dG=5 training pool, filtered the way training filters it
    rows = read_csv(TRAIN_CSV)
    n_raw = len(rows)
    print("dG=5 training pool : %d rows" % n_raw)

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

    # ---- 2) cluster purity: keep a cluster only if all its rows share Source
    by_cluster = {}
    for r in kept:
        by_cluster.setdefault(r["Cluster_Num"], []).append(r)

    subsets = {"PDB": [], "UniProt": []}
    mixed = []
    for cid, members in by_cluster.items():
        srcs = {m["Source"] for m in members}
        if len(srcs) == 1:
            subsets[members[0]["Source"]].extend(members)
        else:
            mixed.extend(members)

    print("\ncluster purity (after pass filter):")
    print("  clusters total   : %d" % len(by_cluster))
    print("  pure PDB clusters: %d  -> %d sequences"
          % (sum(1 for m in by_cluster.values()
                 if len({x['Source'] for x in m}) == 1
                 and m[0]['Source'] == 'PDB'), len(subsets["PDB"])))
    print("  pure UniProt     : %d  -> %d sequences"
          % (sum(1 for m in by_cluster.values()
                 if len({x['Source'] for x in m}) == 1
                 and m[0]['Source'] == 'UniProt'), len(subsets["UniProt"])))
    print("  mixed (dropped)  : %d  -> %d sequences"
          % (sum(1 for m in by_cluster.values()
                 if len({x['Source'] for x in m}) > 1), len(mixed)))

    # ---- 3) invariants I1-I4 before touching any feature file
    pdb_ids = [r["ID"] for r in subsets["PDB"]]
    uni_ids = [r["ID"] for r in subsets["UniProt"]]
    mix_ids = [r["ID"] for r in mixed]
    if set(pdb_ids) & set(uni_ids):
        sys.exit("!! I1 violated: an ID is in both subsets")
    if len(pdb_ids) != len(set(pdb_ids)) or len(uni_ids) != len(set(uni_ids)):
        sys.exit("!! I1 violated: duplicate ID within a subset")
    pdb_cl = {r["Cluster_Num"] for r in subsets["PDB"]}
    uni_cl = {r["Cluster_Num"] for r in subsets["UniProt"]}
    if pdb_cl & uni_cl:
        sys.exit("!! I2 violated: a cluster is in both subsets")
    if any(r["Source"] != "PDB" for r in subsets["PDB"]):
        sys.exit("!! I3 violated: non-PDB row in the PDB subset")
    if any(r["Source"] != "UniProt" for r in subsets["UniProt"]):
        sys.exit("!! I3 violated: non-UniProt row in the UniProt subset")
    for name, rows_ in (("PDB", subsets["PDB"]), ("UniProt", subsets["UniProt"])):
        labs = {r["Label"] for r in rows_}
        if labs != set(CLASSES):
            sys.exit("!! I4 violated: %s subset covers %s, expected %s"
                     % (name, sorted(labs), CLASSES))
    if set(pdb_ids) & set(mix_ids) or set(uni_ids) & set(mix_ids):
        sys.exit("!! I7 violated: a subset ID is also in the mixed dump")
    print("\ninvariants I1-I4, I7 ok")

    def write_csv(path, rows_, cols):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows_:
                w.writerow({c: r[c] for c in cols})

    cols = ["ID", "Sequence", "Label", "Source", "Cluster_Num"]
    write_csv(os.path.join(OUT_DIR, "splits", "pdb.csv"), subsets["PDB"], cols)
    write_csv(os.path.join(OUT_DIR, "splits", "uniprot.csv"), subsets["UniProt"], cols)
    write_csv(os.path.join(OUT_DIR, "splits", "dropped_mixed.csv"), mixed, cols)
    print("wrote splits/{pdb,uniprot,dropped_mixed}.csv")

    for name, rows_ in (("PDB", subsets["PDB"]), ("UniProt", subsets["UniProt"])):
        print("\n%-7s class counts:" % name)
        for c in CLASSES:
            print("    %2d-mer %6d" % (c, sum(1 for r in rows_ if r["Label"] == c)))

    # ---- 4) pull the features for each subset out of the raw matrices
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
        # The raw matrices store the ORIGINAL oligomeric label (1, 2, ... 12).
        # align_data applies LABEL_MAP itself, so the .h5 files must too.
        want = [(r["ID"], r["Label"]) for r in rows_]
        out_i, out_id, out_lab = [], [], []
        missing = bad = 0
        for pid, lab in want:
            ie, is_ = esm_idx.get(pid), sap_idx.get(pid)
            if ie is None or is_ is None:
                missing += 1
                continue
            # I5/I6: the raw matrices must agree with the subset CSV on label
            if int(esm_labs[ie]) != lab or int(sap_labs[is_]) != lab:
                bad += 1
                continue
            out_i.append((ie, is_))
            out_id.append(pid)
            out_lab.append(lab)
        if missing:
            sys.exit("!! I6 violated: %d %s ids absent from the raw feature "
                     "files" % (missing, name))
        if bad:
            sys.exit("!! I5 violated: %d %s rows disagree with the raw "
                     "feature labels" % (bad, name))

        ei = np.array([p[0] for p in out_i], dtype=np.int64)
        si = np.array([p[1] for p in out_i], dtype=np.int64)
        for branch, src_feats, sel in (("seq", esm_feats, ei),
                                       ("struct", sap_feats, si)):
            out = os.path.join(OUT_DIR, "features", "%s_%s.h5" % (key, branch))
            with h5py.File(out, "w") as h:
                h.create_dataset("ids", data=np.array(out_id, dtype=object),
                                 dtype=h5py.string_dtype(encoding="utf-8"))
                h.create_dataset("labels", data=np.array(out_lab, dtype=np.int64))
                h.create_dataset("features", data=src_feats[sel],
                                 compression="gzip", compression_opts=4)
            print("  %-12s -> %-40s %d x %d"
                  % (branch, os.path.relpath(out, ROOT),
                     len(out_id), src_feats.shape[1]))
    print("\ninvariants I5-I6 ok")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
