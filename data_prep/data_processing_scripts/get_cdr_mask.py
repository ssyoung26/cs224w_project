import os
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

def process_antibody_sequences(input_csv, pdb_folder):
    """
    Processes PDB files to generate antibody_cdr masks based on H1, H2, H3 sequences.

    Parameters:
        input_csv (str): Path to the CSV file containing PDB IDs and CDR sequences.
        pdb_folder (str): Path to the folder containing PDB files.

    Returns:
        dict: Dictionary containing sequences and CDR masks for each PDB ID.
    """
    cdr_data = pd.read_csv(input_csv)
    pdb_ids = cdr_data['PDB_ID']
    h1_sequences = cdr_data['H1']
    h2_sequences = cdr_data['H2']
    h3_sequences = cdr_data['H3']
    parser = PDBParser(QUIET=True)
    antibody_cdr_mapping = {}
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
        antibody_sequence = ""
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id("CA"):  # Only include residues with alpha carbons
                        resname = residue.get_resname()
                        try:
                            antibody_sequence += seq1(resname)
                        except Exception:
                            continue 
        antibody_cdr = [0] * len(antibody_sequence)
        for cdr_seq, cdr_value in [(h1, 1), (h2, 2), (h3, 3)]:
            if pd.notna(cdr_seq):  
                start = antibody_sequence.find(cdr_seq)
                if start != -1:  
                    for i in range(start, start + len(cdr_seq)):
                        antibody_cdr[i] = cdr_value
        antibody_cdr_mapping[pdb_id] = {
            "sequence": antibody_sequence,
            "cdr_mask": antibody_cdr
        }

    return antibody_cdr_mapping
input_csv = "antibody_with_cdr_h123_unique_with-antigen.csv" 
#pdb_folder = "ab_chothia_struct"  
pdb_folder = "ab_test"  

for pdb_id, data in antibody_cdr_mapping.items():
    print(f"PDB ID: {pdb_id}")
    print(f"Sequence: {data['sequence']}")
    print(f"CDR Mask: {data['cdr_mask']}")

    print(len(data['sequence']), len(data['cdr_mask']))
