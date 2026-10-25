#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build SaProt extraction inputs for the predicted-versus-experimental 2x2.

For each split, writes a CSV with one row per protein that has both an
experimental monomer chain and a Chai-1 model.

Usage:
    python build_saprot_inputs.py [--alignments-dir <dir>] [--chains-dir <dir>] [--out-dir <dir>]
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT_DIR = os.path.join(HERE, "features")

SPLITS = ("train", "val")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignments-dir", default=HERE)
    ap.add_argument("--chains-dir", default=os.path.join(HERE, "chains"))
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for sp in SPLITS:
        aln = os.path.join(args.alignments_dir, "alignments_%s.csv" % sp)
        if not os.path.exists(aln):
            sys.exit("missing %s -- run the chain extraction first" % aln)

        rows, missing, checked = [], 0, 0
        with open(aln, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if str(r.get("is_selected", "")).strip() not in ("True", "true", "1"):
                    continue
                pid = r["ID"].strip()
                # real_path is written by extract_chains.py and points at the
                # extracted monomer (chain renamed to A, waters removed).
                rp = (r.get("real_path") or "").strip()
                if not rp:
                    rp = os.path.join(args.chains_dir, sp, "%s.cif" % pid)
                checked += 1
                rows.append({
                    "ID": pid,
                    "Label": r.get("Label", "").strip(),
                    "real_path": rp,
                    "chai_path": (r.get("chai_path") or "").strip(),
                    "pass": str(r.get("pass", "")).strip(),
                    "Source": str(r.get("Source", "")).strip(),
                })

        # Two arms of the 2x2: the same IDs, once pointed at the experimental
        # chain and once at the Chai-1 model that represents them. Holding the
        # ID set fixed is what makes the comparison about coordinates.
        for arm, col, fallback in (
            ("exp",  "real_path", os.path.join(args.chains_dir, sp, "{ID}.cif")),
            ("chai", "chai_path", None),
        ):
            out_rows, arm_missing = [], 0
            for r in rows:
                path = (r.get(col) or "").strip()
                if not path and fallback:
                    path = fallback.format(ID=r["ID"])
                if not path or not os.path.exists(path):
                    arm_missing += 1
                    continue
                out_rows.append({"ID": r["ID"], "Label": r["Label"],
                                 "path": path, "pass": r["pass"],
                                 "Source": r["Source"]})

            out = os.path.join(out_dir, "input_%s%s.csv"
                               % (sp, "" if arm == "exp" else "_chai"))
            with open(out, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, ["ID", "Label", "path", "pass", "Source"])
                w.writeheader()
                w.writerows(out_rows)

            npass = sum(1 for r in out_rows if r["pass"] == "True")
            print("%-6s %-5s  cif missing=%-5d  wrote=%-6d  (pass %d / fail %d)"
                  % (sp, arm, arm_missing, len(out_rows), npass, len(out_rows) - npass))
            print("        -> %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
