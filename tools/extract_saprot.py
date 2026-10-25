# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import torch
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

# Markers that mean "the B-factor in this file is a predicted confidence score
# (pLDDT), so low values should be masked". SaProt's own `auto` only knows the
# first of these, which is why it silently switched masking OFF for Chai-1.
PREDICTED_STRUCTURE_MARKERS = ("alphafold", "chai-1", "chai1", "chai lab",
                               "colabfold", "esmfold", "rosettafold")


def resolve_plddt_mask(plddt_mask, cif_path):
    """Turn --plddt_mask into the bool get_struc_seq expects.

    The real question is "is the B-factor pLDDT?" -- predicted structures
    answer yes (mask), experimental ones answer no (their B-factor is a
    crystallographic displacement factor, where LOW means well ordered).
    """
    if isinstance(plddt_mask, bool) or plddt_mask in (True, False):
        return bool(plddt_mask)
    if str(plddt_mask).lower() == "true":
        return True
    if str(plddt_mask).lower() == "false":
        return False
    # "auto": infer from the file header/title
    try:
        with open(cif_path, "r") as fh:
            head = fh.read(8192).lower()
        hit = next((m for m in PREDICTED_STRUCTURE_MARKERS if m in head), None)
        return hit is not None
    except OSError:
        # Cannot tell -- the SAFE default is to mask, because the released
        # model was trained on masked features. Never silently unmask.
        return True


def get_saprot_embedding(model, tokenizer, device, foldseek_bin, cif_path, chain_id="A",
                         plddt_mask="auto"):
    from utils.foldseek_util import get_struc_seq
    try:
        plddt_mask = resolve_plddt_mask(plddt_mask, cif_path)
        parsed_seqs = get_struc_seq(foldseek_bin, cif_path, [chain_id],
                                    plddt_mask=plddt_mask)
        
        if parsed_seqs is None or len(parsed_seqs) == 0:
            return None
            
        if chain_id in parsed_seqs:
            seq, foldseek_seq, combined_seq = parsed_seqs[chain_id]
        else:
            first_chain = list(parsed_seqs.keys())[0]
            seq, foldseek_seq, combined_seq = parsed_seqs[first_chain]

        inputs = tokenizer(combined_seq, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = model.get_hidden_states(inputs, reduction="mean")
            feat = embeddings[0].squeeze(0)
            return feat.float().cpu().numpy()
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser(description="SaProt Feature Extraction")
    
    parser.add_argument("--config", type=str, default=None, help="Path to JSON configuration file")
    
    parser.add_argument("--csv", default=None, help="Path to input CSV file")
    parser.add_argument("--structure_dir", default=None, help="Directory containing CIF structure files")
    parser.add_argument("--out", default=None, help="Path to output .h5 features file")
    parser.add_argument("--model_path", default=None, help="Path to SaProt model weights (including config.json)")
    parser.add_argument("--saprot_repo", default=None, help="SaProt repository root directory")
    parser.add_argument("--foldseek_bin", default=None, help="Path to Foldseek binary")
    parser.add_argument("--plddt_mask", default="true", choices=("true", "false", "auto"),
                        help="Mask 3Di tokens where the B-factor is below 70. The "
                             "question this switch answers is 'is the B-factor "
                             "actually pLDDT?'. 'true' (DEFAULT) always masks -- "
                             "correct for PREDICTED structures (Chai-1, AlphaFold, "
                             "...), whose B-factor IS pLDDT and where a low score "
                             "means the 3Di token is unreliable. 'false' never "
                             "masks -- REQUIRED for EXPERIMENTAL coordinates "
                             "(X-ray, cryo-EM), where the B-factor is a "
                             "crystallographic displacement factor and LOW is GOOD, "
                             "so masking would blank the whole 3Di channel. 'auto' "
                             "infers the case from the file text (SaProt's own rule "
                             "only recognises AlphaFold and therefore silently "
                             "disables masking for Chai-1 -- prefer an explicit "
                             "value).")
    args = parser.parse_args()
    # argparse keeps the spelling; get_struc_seq wants a real bool or "auto"
    _mask = {"true": True, "false": False, "auto": "auto"}[args.plddt_mask.lower()]

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_args = json.load(f)
        
        cli_args = [arg.lstrip('-').split('=')[0] for arg in sys.argv[1:]]
        
        for key, value in config_args.items():
            if key not in cli_args:
                setattr(args, key, value)

    required_args = ['csv', 'out', 'model_path', 'saprot_repo', 'foldseek_bin']
    missing_args = [arg for arg in required_args if getattr(args, arg) is None]
    if missing_args:
        parser.error(f"Missing required arguments: {', '.join(['--' + arg for arg in missing_args])} must be specified via CLI or config file.")

    sys.path.append(args.saprot_repo)
    try:
        from model.saprot.base import SaprotBaseModel
        from transformers import EsmTokenizer
    except ImportError as e:
        print(f"Error: Import failed ({e})")
        print(f"Please ensure {args.saprot_repo} is the root directory of SaProt.")
        sys.exit(1)

    if not os.path.exists(args.foldseek_bin):
        print(f"Error: Foldseek not found at: {args.foldseek_bin}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SaProt (PDB version) to {device}...")
    
    config = {
        "task": "base",
        "config_path": args.model_path, 
        "load_pretrained": True,
    }
    
    model = SaprotBaseModel(**config)
    tokenizer = EsmTokenizer.from_pretrained(config["config_path"])
    
    if device.type == 'cuda':
        model.half() 
    model.to(device)
    model.eval()

    print(f"\nProcessing dataset from {args.csv}...")
    df = pd.read_csv(args.csv)
    df['ID'] = df['ID'].astype(str).str.strip()
    
    ids_list, labels_list, features_list = [], [], []
    print(f"pLDDT masking    : {args.plddt_mask} -> {_mask}")
    print(f"sequences        : {len(df)}")
    pbar = tqdm(df.iterrows(), total=len(df), desc="Extracting SaProt")
    
    for idx, row in pbar:
        prot_id = row['ID']
        label = row['Label'] if 'Label' in row else row.get('label', -1)

        # The CSV may carry a `path` column naming the exact Chai-1 candidate
        # chosen for this ID (select_structures.py writes it); use it when it
        # is present. Otherwise fall back to the flat <structure_dir>/<ID>.cif
        # layout.
        cif_path = row.get('path') if 'path' in row else None
        if not isinstance(cif_path, str) or not cif_path.strip():
            if args.structure_dir is None:
                sys.exit(f"{prot_id}: the CSV has no usable `path` column and "
                         f"--structure_dir was not given, so there is nothing to "
                         f"fall back to.")
            cif_path = os.path.join(args.structure_dir, f"{prot_id}.cif")
        else:
            cif_path = cif_path.strip()
        
        emb = get_saprot_embedding(model, tokenizer, device, args.foldseek_bin, cif_path,
                                   chain_id="A", plddt_mask=_mask)
        
        if emb is not None:
            ids_list.append(prot_id)
            labels_list.append(label)
            features_list.append(emb)
        else:
            pbar.write(f"Failed: {prot_id}")

    if len(ids_list) > 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        
        features_array = np.vstack(features_list)
        labels_array = np.array(labels_list, dtype=np.int64)
        
        print(f"Saving to HDF5 format: {args.out}")
        print(f"   Shape: {features_array.shape}") 

        with h5py.File(args.out, 'w') as f:
            dt_str = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('ids', data=np.array(ids_list, dtype=object), dtype=dt_str)
            f.create_dataset('labels', data=labels_array)
            f.create_dataset('features', data=features_array, compression='gzip', compression_opts=4)
            
            f.attrs['n_sequences'] = len(ids_list)
            f.attrs['feature_dim'] = features_array.shape[1]

        print("Done! Features saved.")
    else:
        print("No features extracted.")

if __name__ == "__main__":
    main()