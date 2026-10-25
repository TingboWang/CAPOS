#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute TM-scores between Chai-1 models and experimental assembly chains.

For PDB-derived proteins, selects the best-matching experimental chain and
writes RMSD, TM-score, coverage, and alignment length per protein.

Usage:
    python compute_tm.py --base <CAPOS root> --split <val|test> --out <csv> --usalign <US-align binary>
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

USALIGN = None
SELECTION = None
SPLITS = None
ASM_DIR = None

# A chain counts as protein if it has at least this many standard residues.
MIN_AA = 10
# A chain counts as the query's own copy if at least this fraction of the query
# is aligned. Keeps an unrelated small protein in the same assembly from
# winning on a lucky local match.
MIN_COVERAGE = 0.5

AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}

OUTFMT2_HEADER = ["PDBchain1", "PDBchain2", "TM1", "TM2", "RMSD", "ID1",
                  "ID2", "IDali", "L1", "L2", "Lali"]


def protein_chains(cif_path):
    """(chain_name, n_standard_aa) for every protein chain of the assembly."""
    import gemmi

    st = gemmi.read_structure(cif_path)
    if len(st) == 0:
        return []
    out = []
    for ch in st[0]:
        n_aa = sum(1 for r in ch if r.name in AA3)
        if n_aa >= MIN_AA:
            out.append((ch.name, n_aa))
    return out


def extract_chain(cif_path, chain_name, out_path):
    """Write just one chain of the assembly to its own mmCIF."""
    import gemmi

    st = gemmi.read_structure(cif_path)
    model = st[0].clone()
    for ch in list(model):
        if ch.name != chain_name:
            model.remove_chain(ch.name)
    sub = gemmi.Structure()
    sub.add_model(model)
    sub.make_mmcif_document().write_file(out_path)
    return out_path


def usalign(pred_cif, chain_cif, workdir, usalign_bin):
    """One US-align call. Returns the outfmt-2 row as a dict, or None."""
    r = subprocess.run(
        [usalign_bin, pred_cif, chain_cif, "-mol", "prot", "-outfmt", "2"],
        capture_output=True, text=True, cwd=workdir)
    for line in r.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < len(OUTFMT2_HEADER):
            continue
        d = dict(zip(OUTFMT2_HEADER, f))
        try:
            d["TM1"] = float(d["TM1"])
            d["TM2"] = float(d["TM2"])
            d["RMSD"] = float(d["RMSD"])
            d["IDali"] = float(d["IDali"])
            d["L1"] = int(d["L1"])
            d["L2"] = int(d["L2"])
            d["Lali"] = int(d["Lali"])
        except (TypeError, ValueError):
            return None
        return d
    return None


def score_one(row, workdir, usalign_bin):
    """Score a single protein. Returns a result dict (never raises)."""
    prot_id = row["ID"]
    pred_cif = row["path"]
    asm_cif = os.path.join(ASM_DIR, "%s-assembly1.cif" % prot_id)

    res = {
        "ID": prot_id,
        "Label": row.get("Label", ""),
        "Source": row.get("Source", ""),
        "best_ptm": row.get("best_ptm", ""),
        "best_plddt": row.get("best_plddt", ""),
        "n_chains": 0,
        "best_chain": "",
        "tm_score": "",
        "tm_score_chain": "",
        "rmsd": "",
        "seq_id": "",
        "aligned_len": "",
        "len_pred": "",
        "len_chain": "",
        "per_chain": "",
        "status": "",
    }

    for p in (pred_cif, asm_cif):
        if not p or not os.path.exists(p):
            res["status"] = "missing_file"
            return res

    try:
        chains = protein_chains(asm_cif)
    except Exception:
        res["status"] = "assembly_unreadable"
        return res

    res["n_chains"] = len(chains)
    if not chains:
        res["status"] = "no_protein_chain"
        return res

    per_chain = []
    candidates = []
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        for name, n_aa in chains:
            chain_cif = os.path.join(td, "chain_%s.cif" % name.replace("/", "_"))
            try:
                extract_chain(asm_cif, name, chain_cif)
            except Exception:
                per_chain.append({"chain": name, "status": "extract_failed"})
                continue
            d = usalign(pred_cif, chain_cif, td, usalign_bin)
            if d is None:
                per_chain.append({"chain": name, "status": "usalign_failed"})
                continue
            per_chain.append({
                "chain": name,
                "status": "ok",
                "TM1": round(d["TM1"], 5),
                "TM2": round(d["TM2"], 5),
                "RMSD": d["RMSD"],
                "IDali": round(d["IDali"], 4),
                "Lali": d["Lali"],
                "L1": d["L1"],
                "L2": d["L2"],
            })
            if d["Lali"] >= MIN_COVERAGE * d["L1"]:
                candidates.append((d, name))

    res["per_chain"] = ";".join(
        "%s:%s" % (c["chain"], c.get("TM1", c["status"])) for c in per_chain)

    if not candidates:
        res["status"] = "no_chain_covered_query"
        return res

    # highest TM-score normalised by the query length wins
    d, name = max(candidates, key=lambda t: t[0]["TM1"])
    res.update({
        "best_chain": name,
        "tm_score": round(d["TM1"], 5),
        "tm_score_chain": round(d["TM2"], 5),
        "rmsd": round(d["RMSD"], 3),
        "seq_id": round(d["IDali"], 4),
        "aligned_len": d["Lali"],
        "len_pred": d["L1"],
        "len_chain": d["L2"],
        "status": "ok",
    })
    return res


def main():
    global USALIGN, SELECTION, SPLITS, ASM_DIR
    ap = argparse.ArgumentParser(
        description="TM-score of the Chai-1 prediction against the "
                    "experimental assembly, best protein chain.")
    ap.add_argument("--base", required=True, help="CAPOS repository root")
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N proteins (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--usalign", required=True, help="US-align binary")
    args = ap.parse_args()

    USALIGN = args.usalign
    SELECTION = os.path.join(args.base, "monomer_structures",
                             "structure_selection.csv")
    SPLITS = os.path.join(args.base, "dataset", "dataset_split", "splits")
    ASM_DIR = os.path.join(args.base, "pdb_pisa", "assembly1_cif_gz")

    import pandas as pd

    split_csv = os.path.join(SPLITS, "%s.csv" % args.split)
    split = pd.read_csv(split_csv, encoding="utf-8-sig")
    sel = pd.read_csv(SELECTION, encoding="utf-8-sig")

    # only the PDB-sourced, structure-passing rows -- UniProt entries have no
    # experimental structure, and the pass filter is the same one that decided
    # which proteins entered the training and evaluation sets at all.
    rows = split.merge(
        sel[["ID", "pass", "path", "best_idx", "best_ptm", "best_plddt"]],
        on="ID", how="inner")
    rows = rows[(rows["Source"] == "PDB") & (rows["pass"] == True)]  # noqa: E712
    if args.limit:
        rows = rows.head(args.limit)

    print("split      : %s" % args.split)
    print("universe   : %d  (PDB-sourced and structure-passing)" % len(rows))
    print("usalign    : %s" % args.usalign)

    workdir = os.path.join(HERE, "work_%s" % args.split)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    jobs = rows.to_dict("records")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(score_one, r, workdir, args.usalign) for r in jobs]
        for i, f in enumerate(futs, 1):
            results.append(f.result())
            if i % 500 == 0 or i == len(futs):
                ok = sum(1 for r in results if r["status"] == "ok")
                print("  %d/%d  ok=%d" % (i, len(futs), ok), flush=True)

    fields = ["ID", "Label", "Source", "best_ptm", "best_plddt", "n_chains",
              "best_chain", "tm_score", "tm_score_chain", "rmsd", "seq_id",
              "aligned_len", "len_pred", "len_chain", "status", "per_chain"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)

    ok = sum(1 for r in results if r["status"] == "ok")
    print()
    print("wrote %s  (%d rows, %d scored)" % (args.out, len(results), ok))
    from collections import Counter
    for k, v in Counter(r["status"] for r in results).most_common():
        print("  %-24s %d" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
