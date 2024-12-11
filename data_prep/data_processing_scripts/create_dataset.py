import os
import json
import random
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

def extract_features_and_save_splits(input_csv, pdb_folder, output_folder):
    """
    Processes PDB files to extract antibody and antigen features, splits into train/test/val, and saves to JSONL.

    Parameters:
        input_csv (str): Path to the CSV file containing PDB IDs and CDR sequences.
        pdb_folder (str): Path to the folder containing PDB files.
        output_folder (str): Path to save the output JSONL files.
    """
    cdr_data = pd.read_csv(input_csv)
    pdb_ids = cdr_data['PDB_ID']
    h1_sequences = cdr_data['H1']
    h2_sequences = cdr_data['H2']
    h3_sequences = cdr_data['H3']

    parser = PDBParser(QUIET=True)

    pdb_features = {}

    for pdb_id, h1, h2, h3 in zip(pdb_ids, h1_sequences, h2_sequences, h3_sequences):
        pdb_file = os.path.join(pdb_folder, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_file):
            print(f"PDB file not found: {pdb_file}")
            continue

        try:
            structure = parser.get_structure(pdb_id, pdb_file)
        except Exception as e:
            print(f"Error parsing {pdb_file}: {e}")
            continue

        antibody_seq, antigen_seq = "", ""
        antibody_coords, antigen_coords = [], []
        antibody_atypes, antigen_atypes = [], []

        for model in structure:
            for chain in model:
                chain_seq = ""
                chain_coords = []
                chain_atypes = []

                for residue in chain:
                    if residue.has_id("CA"):
                        resname = residue.get_resname()
                        try:
                            chain_seq += seq1(resname)
                        except Exception:
                            continue 

                    for atom in residue:
                        chain_coords.append(list(atom.coord))
                        chain_atypes.append(ord(atom.element[0]) % 20)  

                if h1 in chain_seq or h2 in chain_seq or h3 in chain_seq:
                    antibody_seq += chain_seq
                    antibody_coords.extend(chain_coords)
                    antibody_atypes.extend(chain_atypes)
                else:
                    antigen_seq += chain_seq
                    antigen_coords.extend(chain_coords)
                    antigen_atypes.extend(chain_atypes)

        pdb_features[pdb_id] = {
            "antibody_seq": antibody_seq,
            "antibody_coords": antibody_coords,
            "antibody_atypes": antibody_atypes,
            "antigen_seq": antigen_seq,
            "antigen_coords": antigen_coords,
            "antigen_atypes": antigen_atypes
        }

    pdb_ids_list = list(pdb_features.keys())
    random.shuffle(pdb_ids_list)

    train_split = int(0.8 * len(pdb_ids_list))
    val_split = int(0.9 * len(pdb_ids_list))

    train_ids = pdb_ids_list[:train_split]
    val_ids = pdb_ids_list[train_split:val_split]
    test_ids = pdb_ids_list[val_split:]

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }

    os.makedirs(output_folder, exist_ok=True)
    
    for split_name, split_ids in splits.items():
        split_file = os.path.join(output_folder, f"{split_name}_data.jsonl")
        with open(split_file, 'w') as jsonl_file:
            for pdb_id in split_ids:
                json.dump({pdb_id: pdb_features[pdb_id]}, jsonl_file)
                jsonl_file.write('\n')
        print(f"{split_name.capitalize()} data saved to {split_file}")

input_csv = "antibody_with_cdr_h123_unique_with-antigen.csv"  
pdb_folder = "ab_struct"  
output_folder = "ab_fold_processed" 

extract_features_and_save_splits(input_csv, pdb_folder, output_folder)

