import os
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

def extract_features_from_pdb(input_csv, pdb_folder):
    """
    Processes PDB files to extract antibody and antigen features.

    Parameters:
        input_csv (str): Path to the CSV file containing PDB IDs and CDR sequences.
        pdb_folder (str): Path to the folder containing PDB files.

    Returns:
        dict: Dictionary containing antibody and antigen features for each PDB ID.
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

    return pdb_features

input_csv = "antibody_with_cdr_h123_unique_with-antigen.csv" 
pdb_folder = "sample_data/ab_struct"  
pdb_features = extract_features_from_pdb(input_csv, pdb_folder)
for pdb_id, features in pdb_features.items():
    print(f"PDB ID: {pdb_id}")
    for key, value in features.items():
        if isinstance(value, list) and len(value) > 10:  
            print(f"{key}: {value[:10]} ... ({len(value)} items)")
        else:
            print(f"{key}: {value}")

