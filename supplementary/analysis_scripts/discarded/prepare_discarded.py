#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the structure-quality-excluded proteins for scoring.

Collects the proteins removed by the pTM/pLDDT gate and writes the inputs
needed to score them with CAPOS.

Usage:
    python prepare_discarded.py [--base <CAPOS root>] [--selection <csv>] [--splits-dir <dir>] [--cluster-membership <csv>] [--out_dir <dir>]
"""
import argparse
import csv
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def find_root(start):
    """Walk up to the CAPOS root, so the script works at any nesting depth."""
    p = os.path.abspath(start)
    while p != os.path.dirname(p):
        if (os.path.isdir(os.path.join(p, "model"))
                and os.path.isdir(os.path.join(p, "benchmark"))):
            return p
        p = os.path.dirname(p)
    raise SystemExit("could not locate the CAPOS root above %s" % start)


ROOT = None
SELECTION = None
SPLITS_DIR = None
CLUSTER_MEMBERSHIP = None


def main():
    global ROOT, SELECTION, SPLITS_DIR, CLUSTER_MEMBERSHIP
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="CAPOS repository root")
    ap.add_argument("--selection", default=None)
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--cluster-membership", default=None)
    ap.add_argument("--out_dir", default=HERE,
                    help="where the staged CSVs land (default: this script's folder)")
    args = ap.parse_args()

    if args.base:
        ROOT = os.path.abspath(args.base)
    else:
        ROOT = find_root(HERE)
    SELECTION = os.path.join(ROOT, "monomer_structures/structure_selection.csv")
    SPLITS_DIR = os.path.join(ROOT, "dataset/dataset_split/splits")
    # Clustering was run once on the raw corpus, so every sequence has a
    # Cluster_Num even when the dG=5 filter removes it from the training split.
    CLUSTER_MEMBERSHIP = os.path.join(ROOT, "dataset/cluster/cluster_membership.csv")
    if args.selection:
        SELECTION = args.selection
    if args.splits_dir:
        SPLITS_DIR = args.splits_dir
    if args.cluster_membership:
        CLUSTER_MEMBERSHIP = args.cluster_membership
    args.selection = SELECTION

    # ---- what the filter dropped --------------------------------------------
    sel, sel_fields = {}, None
    with open(args.selection, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if sel_fields is None:
                sel_fields = list(r.keys())
            sel[r["ID"].strip()] = r

    failed = [i for i, r in sel.items() if r["pass"].strip() != "True"]
    print("pool rows         : %d" % len(sel))
    print("dropped by the cut: %d  (%.2f%%)"
          % (len(failed), 100.0 * len(failed) / len(sel)))

    # ---- cluster membership across all three splits -------------------------
    # Split CSVs are written before the structure filter, so a sequence can be
    # listed in dataset_5_train.csv and still be excluded from training. Only
    # the pass-filtered training subset counts as "seen".
    #
    # Cluster IDs are taken from the dG=5 split files where available. For
    # sequences removed by the dG=5 filter, fall back to cluster_membership.csv
    # so their homology exposure is not treated as "no cluster".
    all_cluster, train_ids = {}, set()
    for name in ("dataset_5_train", "val", "test"):
        p = os.path.join(SPLITS_DIR, "%s.csv" % name)
        if not os.path.exists(p):
            print("  !! missing %s" % p)
            continue
        with open(p, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                cid, cl = r["ID"].strip(), r["Cluster_Num"].strip()
                all_cluster[cid] = cl
                if name == "dataset_5_train":
                    train_ids.add(cid)
    print("split rows read   : %d  (training-pool rows: %d)"
          % (len(all_cluster), len(train_ids)))

    raw_cluster = {}
    if os.path.exists(CLUSTER_MEMBERSHIP):
        with open(CLUSTER_MEMBERSHIP, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                raw_cluster[r["ID"].strip()] = r["Cluster_Num"].strip()
        print("raw cluster map   : %d  (%s)"
              % (len(raw_cluster), os.path.basename(CLUSTER_MEMBERSHIP)))
    else:
        print("  !! missing %s -- dG-filtered IDs will have no cluster"
              % CLUSTER_MEMBERSHIP)

    def cluster_of(cid):
        cl = all_cluster.get(cid, "")
        if cl:
            return cl
        return raw_cluster.get(cid, "")

    # only rows that cleared the filter were trained on
    kept_train = {i for i in train_ids if sel.get(i, {}).get("pass", "").strip() == "True"}
    train_clusters = {cluster_of(i) for i in kept_train if cluster_of(i)}
    print("trained on (after the filter): %d rows, %d clusters"
          % (len(kept_train), len(train_clusters)))
    in_train_pool = [i for i in failed if i in train_ids]
    print("  discarded IDs that were themselves in the training CSV: %d "
          "(dropped by the filter, never trained on)"
          % len(in_train_pool))

    n_known = sum(1 for i in failed if cluster_of(i))
    shared = [i for i in failed
              if cluster_of(i) and cluster_of(i) in train_clusters
              and i not in kept_train]
    print()
    print("discarded with a known cluster : %d / %d" % (n_known, len(failed)))
    print("  cluster contains a TRAINED-ON sequence: %d (%.1f%%)"
          % (len(shared),
             100.0 * len(shared) / n_known if n_known else 0))
    print("  -> those are not strictly unseen: a homologue was trained on")
    print("  -> %d have no trained-on homologue"
          % (n_known - len(shared)))

    # ---- write the two inputs ----------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    p_split = os.path.join(args.out_dir, "discarded_split.csv")
    with open(p_split, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Sequence", "Label", "Cluster_Num"])
        for i in sorted(failed):
            r = sel[i]
            w.writerow([i, r["Sequence"].strip(), r["Label"].strip(),
                        cluster_of(i)])
    print()
    print("wrote %s  (%d rows)" % (p_split, len(failed)))

    p_sel = os.path.join(args.out_dir, "discarded_selection.csv")
    with open(p_sel, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(sel_fields)
        for i in sorted(failed):
            row = dict(sel[i])
            row["pass"] = "True"          # the point of the file
            w.writerow([row[k] for k in sel_fields])
    print("wrote %s  (%d rows, all pass=True)" % (p_sel, len(failed)))

    p_prof = os.path.join(args.out_dir, "discarded_profile.csv")
    with open(p_prof, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Label", "length", "best_ptm", "best_plddt",
                    "cluster", "cluster_has_trained_on"])
        for i in sorted(failed):
            r = sel[i]
            c = cluster_of(i)
            w.writerow([i, r["Label"].strip(), len(r["Sequence"].strip()),
                        r["best_ptm"], r["best_plddt"], c,
                        "yes" if (c and c in train_clusters) else "no"])
    print("wrote %s" % p_prof)

    print()
    print("label mix of the dropped:")
    c = Counter(int(sel[i]["Label"]) for i in failed)
    for k in sorted(c):
        print("  label %-3d %6d  (%.1f%%)" % (k, c[k], 100.0 * c[k] / len(failed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
