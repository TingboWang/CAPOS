# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import argparse
import torch
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel

def preprocess_sequence(seq):
    return " ".join(list(re.sub(r"[UZOB]", "X", seq.upper())))

def extract_per_protein_features(df, tokenizer, model, device, batch_size):
    sequences = [preprocess_sequence(seq) for seq in df['Sequence'].tolist()]
    ids_list = df['ID'].astype(str).tolist()
    if 'Label' in df:
        labels_list = df['Label'].tolist()
    elif 'label' in df:
        labels_list = df['label'].tolist()
    else:
        labels_list = [-1] * len(df)
    
    all_features = []
    n_total = len(sequences)

    for start in tqdm(range(0, n_total, batch_size), desc="Extracting ProtT5 features"):
        batch_sequences = sequences[start:start+batch_size]
        
        tokenized = tokenizer(batch_sequences, add_special_tokens=True, padding="longest", return_tensors="pt")
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_embeddings = []
            for i, seq in enumerate(batch_sequences):
                seq_len = len(seq.split())
                emb = outputs.last_hidden_state[i, :seq_len, :].mean(dim=0)
                batch_embeddings.append(emb.cpu().numpy())
            
            all_features.append(np.stack(batch_embeddings, axis=0))

    if all_features:
        features_array = np.vstack(all_features)
        return ids_list, labels_list, features_array
    else:
        return [], [], np.array([])

def main():
    parser = argparse.ArgumentParser(description="ProtT5 Feature Extraction")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON configuration file")
    parser.add_argument("--csv", default=None, help="Path to input CSV file")
    parser.add_argument("--out", default=None, help="Path to output .h5 features file")
    parser.add_argument("--model_name", type=str, default="Rostlab/prot_t5_xl_half_uniref50-enc", help="Model name or local path")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_args = json.load(f)
        
        cli_args = [arg.lstrip('-').split('=')[0] for arg in sys.argv[1:]]
        
        for key, value in config_args.items():
            if key not in cli_args:
                setattr(args, key, value)

    required_args = ['csv', 'out']
    missing_args = [arg for arg in required_args if getattr(args, arg) is None]
    if missing_args:
        parser.error(f"Missing required arguments: {', '.join(['--' + arg for arg in missing_args])} must be specified via CLI or config file.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ProtT5 tokenizer and model to {device}...")
    
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(args.model_name).to(device)
    model.eval()
    model.to(torch.float32)

    print(f"\nProcessing dataset: {args.csv}")
    df = pd.read_csv(args.csv).dropna(subset=["Sequence"])
    ids_list, labels_list, features_array = extract_per_protein_features(df, tokenizer, model, device, args.batch_size)
    raw_sequences = df["Sequence"].astype(str).tolist()

    if features_array.size > 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        
        with h5py.File(args.out, 'w') as f:
            dt_str = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('ids', data=np.array(ids_list, dtype=object), dtype=dt_str)
            f.create_dataset('labels', data=np.array(labels_list, dtype=np.int64))
            f.create_dataset('sequences', data=np.array(raw_sequences, dtype=object), dtype=dt_str)
            f.create_dataset('features', data=features_array, compression='gzip', compression_opts=4)
            
            f.attrs['n_sequences'] = len(ids_list)
            f.attrs['feature_dim'] = features_array.shape[1]

        print(f"Saved per-protein embeddings to HDF5: {args.out}")
    else:
        print("Skipping save: no features extracted.")

if __name__ == "__main__":
    main()