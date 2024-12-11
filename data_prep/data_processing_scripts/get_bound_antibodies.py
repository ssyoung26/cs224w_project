import pandas as pd
import os
import shutil
import json
from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqUtils import seq1

""" Create antibody + cdr dataset from sabdab database. Filter the selection to only
antibodies with antigen bound in the pdb file. """
def create_antibody_folder_from_sabdab():
    summary_file_path = '20241107_0591212_summary.tsv'
    summary_data = pd.read_csv(summary_file_path, sep='\t')
    pdb_list = summary_data.iloc[:, 0].tolist()  # 7190 unique antibody-antigens and cdrs
    cwd = os.getcwd()
    pdb_folder_path = os.path.join(cwd, 'chothia_cdr')
    destination_folder_path = os.path.join(cwd, 'ab_cdr') # or ab_struct
    os.makedirs(destination_folder_path, exist_ok=True)

    for pdb_file in pdb_list:
        pdb_file += ".pdb" 
        src_file_path = os.path.join(pdb_folder_path, pdb_file)
        dest_file_path = os.path.join(destination_folder_path, pdb_file)
        if os.path.exists(src_file_path):  
            shutil.move(src_file_path, dest_file_path)
    return

def main():
        
    pdb_file = "1a14.pdb"
    output_file = "1a14_test.json"

    create_antibody_folder_from_sabdab()

#    for pdb_file in pdb_list:
#        pdb_file += ".pdb" 
#        src_file_path = os.path.join(pdb_folder_path, pdb_file)
#        dest_file_path = os.path.join(destination_folder_path, pdb_file)
#        if os.path.exists(src_file_path):  
#            shutil.move(src_file_path, dest_file_path)
#    return


if __name__ == "__main__":
    main()

