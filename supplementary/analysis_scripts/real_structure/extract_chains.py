#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract experimental monomer chains from PDB biological assemblies.

For each PDB-sourced protein, selects a chain that matches the dataset
sequence and writes it as a single-chain mmCIF for the 2x2 experiment.

Usage:
    python extract_chains.py --base <CAPOS root> --split <train|val|test> --usalign <US-align binary> [--workers N]
"""
import argparse
import csv
import difflib
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "..", "tm_score"))

from compute_tm import (AA3, MIN_AA, protein_chains, usalign)  # noqa: E402

USALIGN = None
ASM_DIR = None
SELECTION = None
SPLIT_DIR = os.path.join(HERE, "splits")
CHAIN_DIR = os.path.join(HERE, "chains")

AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}

MIN_IDENTITY = 0.95        # a chain must be this much the query protein
MIN_COVERAGE = 0.50        # and must cover this much of it

CSV_COLS = ["split", "ID", "Label", "Source", "pass", "best_idx", "best_ptm",
            "best_plddt", "chai_path", "real_path", "n_chains", "chain",
            "is_selected", "seq_identity", "seq_coverage", "TM1", "TM2",
            "RMSD", "IDali", "Lali", "len_pred", "len_chain", "reason"]


# ---------------------------------------------------------------- sequences
def chain_sequence(cif_path, chain_name):
    """One-letter sequence of one chain, standard residues only."""
    import gemmi

    st = gemmi.read_structure(cif_path)
    if len(st) == 0:
        return ""
    for ch in st[0]:
        if ch.name == chain_name:
            return "".join(AA1[r.name] for r in ch if r.name in AA3)
    return ""


def seq_match(query, target):
    """(identity, coverage) of `target` against `query`, from matching blocks.

    identity  = matched residues / length of the shorter sequence
    coverage  = matched residues / length of the query
    """
    if not query or not target:
        return 0.0, 0.0
    sm = difflib.SequenceMatcher(None, query, target, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    ident = matched / min(len(query), len(target))
    cover = matched / len(query)
    return ident, cover


# ------------------------------------------------------------------ extract
def extract_chain_as_a(cif_path, chain_name, out_path):
    """Write one chain of the assembly to its own mmCIF, chain id renamed to A.

    Only standard amino acids are kept. The crystallographic chain carries
    waters and other hetero atoms -- 1A04's chain is 205 residues of protein
    plus 176 waters -- and those would sit in the file that Foldseek reads to
    produce the 3Di sequence. The polymer is what SaProt needs, so the export
    is restricted to it.

    The chain is rebuilt residue by residue rather than filtered in place:
    gemmi's `remove_residue` takes a seqid, and waters share seqids with the
    amino acids they are numbered alongside, so removing by seqid takes the
    polymer with it.
    """
    import gemmi

    st = gemmi.read_structure(cif_path)
    model = st[0].clone()
    for ch in list(model):
        if ch.name != chain_name:
            model.remove_chain(ch.name)
    if len(model) == 0:
        raise ValueError("chain %s not found" % chain_name)

    src = model[0]
    new_ch = gemmi.Chain("A")
    for res in src:
        if res.name in AA3:
            new_ch.add_residue(res.clone())
    if len(new_ch) == 0:
        raise ValueError("chain %s has no amino acids" % chain_name)

    out_model = gemmi.Model("1")
    out_model.add_chain(new_ch)
    sub = gemmi.Structure()
    sub.add_model(out_model)
    sub.make_mmcif_document().write_file(out_path)
    return out_path


def _reject_reason(rows):
    """Which cut the best available chain failed.

    A chain is kept only when it is >= MIN_IDENTITY identical to the query AND
    covers >= MIN_COVERAGE of it. When nothing qualifies, the interesting fact
    is which of the two stopped it -- the query sequence usually matches the
    resolved chain just fine, and what is missing is the rest of the protein
    (crystallographic construct boundaries, unresolved termini, proteolysis).
    Reporting that as an identity failure is wrong.
    """
    measured = [r for r in rows
                if r.get("seq_identity") not in ("", None)
                and r.get("seq_coverage") not in ("", None)]
    if not measured:
        return "no_candidate_chain"
    best = max(measured, key=lambda r: (float(r["seq_identity"]),
                                        float(r["seq_coverage"])))
    id_ok = float(best["seq_identity"]) >= MIN_IDENTITY
    cov_ok = float(best["seq_coverage"]) >= MIN_COVERAGE
    if id_ok and not cov_ok:
        return "no_chain_passed_coverage"
    if cov_ok and not id_ok:
        return "no_chain_passed_identity"
    return "no_chain_passed_identity_and_coverage"


# --------------------------------------------------------------------- main
def process_one(row, split, workdir):
    """Return (list_of_chain_rows, status). Never raises."""
    pid = row["ID"].strip()
    seq = row["Sequence"].strip()
    chai = (row.get("chai_path") or "").strip()
    asm = os.path.join(ASM_DIR, "%s-assembly1.cif" % pid)

    # 'pass' is a Python keyword, so this has to be a dict literal rather than
    # a dict() call with keyword arguments; the column name is fixed by
    # structure_selection.csv and is kept as is.
    base = {
        "split": split, "ID": pid, "Label": row.get("Label", ""),
        "Source": row.get("Source", ""), "pass": row.get("pass", ""),
        "best_idx": row.get("best_idx", ""), "best_ptm": row.get("best_ptm", ""),
        "best_plddt": row.get("best_plddt", ""), "chai_path": chai,
        "real_path": "", "n_chains": 0, "chain": "", "is_selected": "",
        "seq_identity": "", "seq_coverage": "", "TM1": "", "TM2": "", "RMSD": "",
        "IDali": "", "Lali": "", "len_pred": "", "len_chain": "", "reason": "",
    }

    for p, tag in ((asm, "missing_assembly"), (chai, "missing_chai")):
        if not p or not os.path.exists(p):
            base["reason"] = tag
            return [base], tag

    try:
        chains = protein_chains(asm)
    except Exception:
        base["reason"] = "assembly_unreadable"
        return [base], "assembly_unreadable"

    base["n_chains"] = len(chains)
    if not chains:
        base["reason"] = "no_protein_chain"
        return [base], "no_protein_chain"

    rows = []
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        for ch_name, n_aa in chains:
            r = dict(base)
            r["chain"] = ch_name
            r["len_chain"] = n_aa
            chain_cif = os.path.join(td, "%s_%s.cif" % (pid, ch_name))
            try:
                extract_chain_as_a(asm, ch_name, chain_cif)
                # the export renames the chain to A, so the sequence must be
                # read back under that name -- looking for the original chain
                # id returns "" for every chain that was not already called A
                cseq = chain_sequence(chain_cif, "A")
                ident, cover = seq_match(seq, cseq)
                r["seq_identity"] = round(ident, 4)
                r["seq_coverage"] = round(cover, 4)
            except Exception:
                r["reason"] = "chain_extract_failed"
                rows.append(r)
                continue

            d = usalign(chai, chain_cif, td, USALIGN)
            if d:
                r["TM1"] = round(d["TM1"], 4)
                r["TM2"] = round(d["TM2"], 4)
                r["RMSD"] = round(d["RMSD"], 3)
                r["IDali"] = round(d["IDali"], 4)
                r["Lali"] = d["Lali"]
                r["len_pred"] = d["L1"]
            rows.append(r)

    # ---- selection ----------------------------------------------------
    ok = [r for r in rows
          if r["seq_identity"] != "" and r["seq_identity"] >= MIN_IDENTITY
          and r["seq_coverage"] >= MIN_COVERAGE]
    if not ok:
        # Record the criterion that failed. Most rejected chains match part of
        # the sequence but cover too little of it, so the distinction between
        # identity failure and coverage failure matters.
        reason = _reject_reason(rows)
        for r in rows:
            r["reason"] = reason
        return rows, reason

    def rank(r):
        return (-float(r["seq_coverage"]), -float(r["TM1"] or 0),
                -int(r["len_chain"] or 0))
    ok.sort(key=rank)
    win = ok[0]
    win["is_selected"] = True
    win["reason"] = ("max coverage %.3f" % float(win["seq_coverage"]))
    for r in rows:
        if r is not win and not r["reason"]:
            r["is_selected"] = False
            r["reason"] = "not_selected"

    out_dir = os.path.join(CHAIN_DIR, split)
    os.makedirs(out_dir, exist_ok=True)
    out_cif = os.path.join(out_dir, "%s.cif" % pid)
    try:
        extract_chain_as_a(asm, win["chain"], out_cif)
        win["real_path"] = out_cif
    except Exception:
        win["reason"] = "write_failed"
        return rows, "write_failed"
    return rows, "ok"


def main():
    global USALIGN, ASM_DIR, SELECTION, SPLIT_DIR, CHAIN_DIR
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N rows, for a smoke test")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base", required=True, help="CAPOS repository root")
    ap.add_argument("--usalign", required=True, help="US-align binary")
    ap.add_argument("--splits-dir", default=SPLIT_DIR)
    ap.add_argument("--chains-dir", default=CHAIN_DIR)
    ap.add_argument("--work-dir", default=os.path.join(HERE, "work"))
    ap.add_argument("--alignments-out", default=None)
    args = ap.parse_args()

    USALIGN = args.usalign
    ASM_DIR = os.path.join(args.base, "pdb_pisa", "assembly1_cif_gz")
    SELECTION = os.path.join(args.base, "monomer_structures",
                             "structure_selection.csv")
    SPLIT_DIR = args.splits_dir
    CHAIN_DIR = args.chains_dir

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(CHAIN_DIR, exist_ok=True)

    rows = []
    for tag in ("pass", "fail"):
        p = os.path.join(SPLIT_DIR, args.split, "pdb_%s.csv" % tag)
        if not os.path.exists(p):
            print("!! missing %s -- run make_splits.py first" % p)
            raise SystemExit(1)
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                r["_tag"] = tag
                rows.append(r)
    if args.limit:
        rows = rows[:args.limit]
    print("split=%s  rows=%d" % (args.split, len(rows)))

    out_csv = (args.alignments_out
               or os.path.join(HERE, "alignments_%s.csv" % args.split))
    status = {}
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for rs, st in ex.map(lambda r: process_one(r, args.split, args.work_dir),
                                 rows):
                for r in rs:
                    w.writerow({k: r.get(k, "") for k in CSV_COLS})
                status[st] = status.get(st, 0) + 1

    print("wrote %s" % out_csv)
    print("status:")
    for k, v in sorted(status.items(), key=lambda x: -x[1]):
        print("  %-28s %d" % (k, v))
    n_sel = sum(1 for k in status if k == "ok")
    print("selected structures written under %s/%s/" % (CHAIN_DIR, args.split))


if __name__ == "__main__":
    main()
