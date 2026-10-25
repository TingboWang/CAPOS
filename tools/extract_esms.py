# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

def extract_esms_features(df, model, batch_converter, device, max_len, batch_size):
    ids_list, labels_list, features_list = [], [], []

    seq_tuples = [
        (str(row["ID"]), str(row["Sequence"]).replace(" ", "").strip().upper(), row["Label"] if "Label" in row else row.get("label", -1))
        for _, row in df.iterrows() if len(str(row["Sequence"]).strip()) > 0
    ]

    for i in tqdm(range(0, len(seq_tuples), batch_size), desc="Extracting ESM-S features"):
        batch_slice = seq_tuples[i:i+batch_size]
        from torchdrug import data
        
        proteins, esm_batch_for_tokens = [], []
        current_ids, current_labels = [], []

        for seq_id, seq, label in batch_slice:
            if len(seq) > max_len:
                seq = seq[:max_len]

            proteins.append(data.Protein.from_sequence(seq))
            esm_batch_for_tokens.append((seq_id, seq))
            current_ids.append(seq_id)
            current_labels.append(label)

        if not proteins:
            continue

        batch_protein = data.Protein.pack(proteins).to(device)
        _, _, tokens = batch_converter(esm_batch_for_tokens)
        tokens = tokens.to(device)

        with torch.no_grad():
            out = model(batch_protein, input=tokens)
            
            if isinstance(out, dict) and 'graph_feature' in out:
                embeddings = out["graph_feature"].cpu().numpy()
            elif isinstance(out, torch.Tensor):
                embeddings = out.cpu().numpy()
            else:
                raise TypeError("Model output is neither dict with 'graph_feature' nor Tensor.")

        ids_list.extend(current_ids)
        labels_list.extend(current_labels)
        features_list.append(embeddings)

    if features_list:
        return ids_list, labels_list, np.vstack(features_list)
    else:
        return [], [], np.array([])

def main():
    parser = argparse.ArgumentParser(description="ESM-S Feature Extraction")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON configuration file")
    parser.add_argument("--csv", default=None, help="Path to input CSV file")
    parser.add_argument("--out", default=None, help="Path to output .h5 features file")
    parser.add_argument("--model_dir", default=None, help="Directory containing ESM-S model")
    parser.add_argument("--ckpt", default=None, help="Path to pre-trained weights (.pth)")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_args = json.load(f)
        
        cli_args = [arg.lstrip('-').split('=')[0] for arg in sys.argv[1:]]
        
        for key, value in config_args.items():
            if key not in cli_args:
                setattr(args, key, value)

    required_args = ['csv', 'out', 'model_dir', 'ckpt']
    missing_args = [arg for arg in required_args if getattr(args, arg) is None]
    if missing_args:
        parser.error(f"Missing required arguments: {', '.join(['--' + arg for arg in missing_args])} must be specified via CLI or config file.")

    try:
        from torchdrug import models
    except ImportError:
        raise ImportError("Please ensure torchdrug is installed to extract ESM-S features.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ESM-S model via TorchDrug on {device} ...")
    
    esm_model = models.EvolutionaryScaleModeling(
        path=args.model_dir,
        model="ESM-2-650M",
        readout="mean"
    )
    esm_model.load_state_dict(torch.load(args.ckpt, map_location=device))
    esm_model = esm_model.to(device)
    esm_model.eval()

    batch_converter = esm_model.alphabet.get_batch_converter()

    df = pd.read_csv(args.csv).dropna(subset=["Sequence"])
    print(f"\nProcessing file: {args.csv} (n={len(df)})")

    ids_list, labels_list, features_array = extract_esms_features(df, esm_model, batch_converter, device, args.max_len, args.batch_size)

    if features_array.size > 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        
        with h5py.File(args.out, 'w') as f:
            dt_str = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('ids', data=np.array(ids_list, dtype=object), dtype=dt_str)
            f.create_dataset('labels', data=np.array(labels_list, dtype=np.int64))
            f.create_dataset('features', data=features_array, compression='gzip', compression_opts=4)
            
            f.attrs['n_sequences'] = len(ids_list)
            f.attrs['feature_dim'] = features_array.shape[1]

        print(f"Saved HDF5: {args.out}")
        print(f"   n_sequences={len(ids_list)}, feature_dim={features_array.shape[1]}")
    else:
        print("Skipping save: no features extracted.")

if __name__ == "__main__":
    main()