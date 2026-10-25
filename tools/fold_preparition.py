# -*- coding: utf-8 -*-
import csv
import sys
import os
import argparse

def sanitize_id(raw_id):
    clean = raw_id.strip().lstrip(">")
    for char in [" ", "|", "=", ":", ";", "(", ")", "[", "]", "{", "}", ","]:
        clean = clean.replace(char, "_")
    
    if not clean:
        return "unknown_seq"
    return clean

def sanitize_sequence(seq):
    return seq.strip().replace(" ", "").replace("*", "").upper()

def process_csv(input_path, output_path):
    count = 0
    try:
        with open(input_path, "r", encoding='utf-8-sig') as fin, open(output_path, "w") as fout:
            reader = csv.reader(fin)
            header = next(reader, None)
            
            if header is None:
                print(f"Error: CSV file {input_path} is empty.")
                sys.exit(1)

            header_lower = [h.lower() for h in header]
            try:
                id_idx = header_lower.index("id")
                seq_idx = header_lower.index("sequence")
            except ValueError:
                missing = []
                if "id" not in header_lower: missing.append("ID")
                if "sequence" not in header_lower: missing.append("Sequence")
                print(f"Error: CSV is missing required columns: {', '.join(missing)}")
                print(f"Found columns: {header}")
                sys.exit(1)

            for row in reader:
                if not row or len(row) <= max(id_idx, seq_idx):
                    continue
                
                cid = sanitize_id(row[id_idx])
                cseq = sanitize_sequence(row[seq_idx])
                
                fout.write(f">protein|name={cid}\n{cseq}\n")
                count += 1
    except Exception as e:
        print(f"An error occurred while processing CSV: {e}")
        sys.exit(1)
    return count

def process_fasta(input_path, output_path):
    count = 0
    try:
        with open(input_path, "r") as fin, open(output_path, "w") as fout:
            current_id = ""
            current_seq = []
            
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith(">"):
                    if current_id:
                        fout.write(f">protein|name={current_id}\n{''.join(current_seq)}\n")
                        count += 1
                    
                    current_id = sanitize_id(line)
                    current_seq = []
                else:
                    current_seq.append(sanitize_sequence(line))
            
            if current_id:
                fout.write(f">protein|name={current_id}\n{''.join(current_seq)}\n")
                count += 1
    except Exception as e:
        print(f"An error occurred while processing FASTA: {e}")
        sys.exit(1)
    return count

def main():
    parser = argparse.ArgumentParser(description="Prepare FASTA files for folding (Chai-1 compatible) from CSV or FASTA.")
    parser.add_argument("--input", required=True, help="Path to input .csv or .fasta file")
    parser.add_argument("--output", required=True, help="Path to output .fasta file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    ext = os.path.splitext(args.input)[1].lower()
    
    print(f"Processing input file: {args.input} ...")
    
    if ext == ".csv":
        total = process_csv(args.input, args.output)
    elif ext in [".fasta", ".fa", ".fas"]:
        total = process_fasta(args.input, args.output)
    else:
        print(f"Error: Unsupported file extension '{ext}'. Please provide a .csv or .fasta file.")
        sys.exit(1)

    print(f"Success! Processed {total} sequences.")
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()