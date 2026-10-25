#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build PDB-sourced train and validation subsets for the 2x2 experiment.

Restricts the 5 kcal/mol corpus to proteins with an experimental structure
and writes the split CSVs used by downstream feature extraction.

Usage:
    python make_splits.py [--base <CAPOS root>] [--out-dir <dir>]
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

SELECTION = None
SPLITS = None
MERGED = None
ASM_DIR = None
PRED_DIR = None
SPLIT_FILES = {}
DG5 = None
OUT_DIR = os.path.join(HERE, "splits")

# what the training run actually saw after the structure filter
EXPECTED_PASS = {"val": 14407, "test": 14418}


def fail(msg):
    print("  FAIL  " + msg)
    raise SystemExit("invariant broken: " + msg)


def ok(msg):
    print("  PASS  " + msg)


def main():
    global ROOT, SELECTION, SPLITS, MERGED, ASM_DIR, PRED_DIR, SPLIT_FILES, DG5, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=ROOT, help="CAPOS repository root")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    ROOT = os.path.abspath(args.base)
    OUT_DIR = os.path.abspath(args.out_dir)
    SELECTION = os.path.join(ROOT, "monomer_structures", "structure_selection.csv")
    SPLITS = os.path.join(ROOT, "dataset", "dataset_split", "splits")
    MERGED = os.path.join(ROOT, "dataset", "merged_datasets")
    ASM_DIR = os.path.join(ROOT, "pdb_pisa", "assembly1_cif_gz")
    PRED_DIR = os.path.join(ROOT, "monomer_structures", "structures")
    SPLIT_FILES = {
        "train": os.path.join(SPLITS, "dataset_5_train.csv"),
        "val": os.path.join(SPLITS, "val.csv"),
        "test": os.path.join(SPLITS, "test.csv"),
    }
    DG5 = os.path.join(MERGED, "dataset_5.csv")

    sel = pd.read_csv(SELECTION, encoding="utf-8-sig")
    sel["ID"] = sel["ID"].astype(str).str.strip()
    dg5 = pd.read_csv(DG5, encoding="utf-8-sig")
    dg5["ID"] = dg5["ID"].astype(str).str.strip()
    id5 = set(dg5["ID"])

    frames = {}
    for name, path in SPLIT_FILES.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["ID"] = df["ID"].astype(str).str.strip()
        frames[name] = df

    print("=" * 78)
    print("  INVARIANTS")
    print("=" * 78)

    # I1 / I4
    for name, df in frames.items():
        vc = df["Source"].value_counts()
        if vc.get("PDB", 0) + vc.get("UniProt", 0) != len(df):
            fail("I1 %s: PDB+UniProt != n" % name)
        ok("I1 %s: PDB+UniProt == n  (%d = %d + %d)"
           % (name, len(df), vc.get("PDB", 0), vc.get("UniProt", 0)))

    # I2
    allid = pd.concat([f["ID"] for f in frames.values()], ignore_index=True)
    lhs = set(frames["train"]["ID"]) \
        | (set(frames["val"]["ID"]) & id5) \
        | (set(frames["test"]["ID"]) & id5)
    if lhs != id5:
        fail("I2: train ∪ (val∩d5) ∪ (test∩d5) != dataset_5")
    ok("I2: train ∪ (val∩d5) ∪ (test∩d5) == dataset_5  (%d)" % len(id5))
    if not set(frames["train"]["ID"]) <= id5:
        fail("I2b: train is not a subset of dataset_5")
    ok("I2b: train ⊆ dataset_5")
    extra = set(allid) - id5
    ok("I2c: %d IDs live only in the frozen val/test and not in dataset_5"
       % len(extra))
    ok("     (PDB oligomers the dG=5 cut removed from training; by design)")

    # I3
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ov = set(frames[a]["ID"]) & set(frames[b]["ID"])
        if ov:
            fail("I3: %s ∩ %s = %d" % (a, b, len(ov)))
        ok("I3: %s ∩ %s == 0" % (a, b))

    # I4 / I5 / I7 inputs
    pdb_frames = {}
    for name, df in frames.items():
        m = df.merge(sel[["ID", "pass", "path", "best_idx", "best_ptm",
                          "best_plddt"]], on="ID", how="left")
        if m["pass"].isna().any():
            fail("I4 %s: %d IDs missing from structure_selection"
                 % (name, int(m["pass"].isna().sum())))
        pdb = m[m["Source"] == "PDB"].copy()
        npass = int((pdb["pass"] == True).sum())  # noqa: E712
        nfail = int((pdb["pass"] == False).sum())  # noqa: E712
        if npass + nfail != len(pdb):
            fail("I4 %s: pass+fail != PDB" % name)
        ok("I4 %s: PDB %d == pass %d + fail %d" % (name, len(pdb), npass, nfail))
        if name in EXPECTED_PASS:
            allpass = int((m["pass"] == True).sum())  # noqa: E712
            if allpass != EXPECTED_PASS[name]:
                fail("I5 %s: pass==True %d != expected %d"
                     % (name, allpass, EXPECTED_PASS[name]))
            ok("I5 %s: structure-filtered n == %d (matches training)"
               % (name, allpass))
        pdb_frames[name] = pdb

    # I6
    allrows = pd.concat(frames.values(), ignore_index=True)
    if allrows.groupby("ID")["Sequence"].nunique().max() != 1:
        fail("I6: same ID maps to different sequences")
    if allrows.groupby("Sequence")["Label"].nunique().max() != 1:
        fail("I6: same sequence maps to different labels")
    ok("I6: ID↔Sequence and Sequence↔Label consistent across all splits")

    # I7 -- presence on disk
    for name, pdb in pdb_frames.items():
        for tag, sub in (("pass", pdb[pdb["pass"] == True]),   # noqa: E712
                         ("fail", pdb[pdb["pass"] == False])):  # noqa: E712
            if len(sub) == 0:
                continue
            no_asm = [i for i in sub["ID"]
                      if not os.path.exists(os.path.join(
                          ASM_DIR, "%s-assembly1.cif" % i))]
            no_pred = [p for p in sub["path"]
                       if not (isinstance(p, str) and os.path.exists(p))]
            if no_asm or no_pred:
                fail("I7 %s/%s: %d missing BA1, %d missing Chai-1"
                     % (name, tag, len(no_asm), len(no_pred)))
            ok("I7 %s/%s: %d rows, BA1 and Chai-1 both present"
               % (name, tag, len(sub)))

    # ---- write -----------------------------------------------------------
    print()
    print("=" * 78)
    print("  WRITING splits/")
    print("=" * 78)
    cols = ["ID", "Sequence", "Label", "Source", "Cluster_Num",
            "best_idx", "best_ptm", "best_plddt", "pass"]
    written = []
    for name, pdb in pdb_frames.items():
        for tag, sub in (("pass", pdb[pdb["pass"] == True]),   # noqa: E712
                         ("fail", pdb[pdb["pass"] == False])):  # noqa: E712
            out_dir = os.path.join(OUT_DIR, name)
            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, "pdb_%s.csv" % tag)
            df = sub.copy()
            have = [c for c in cols if c in df.columns]
            df = df[have].copy()
            df["chai_path"] = sub["path"].values
            df["real_path"] = ""   # filled in once the best chain is cut out
            df.to_csv(out, index=False)
            written.append((out, len(df)))
            print("  %-58s %6d rows" % (os.path.relpath(out, ROOT), len(df)))

    print()
    print("wrote %d files under %s"
          % (len(written), os.path.relpath(OUT_DIR, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
