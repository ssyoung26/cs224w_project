import pandas as pd

def filter_first_unique_pdb(input_file, pdb_list_file, output_file):
    """
    Filters the dataset to retain only the first occurrence of each unique PDB_ID
    from the provided list, ignoring all subsequent duplicates.

    Parameters:
        input_file (str): Path to the input CSV file.
        pdb_list_file (str): Path to the TSV file containing the PDB list (first column).
        output_file (str): Path to save the filtered dataset.
    """
    df = pd.read_csv(input_file)
    pdb_list_df = pd.read_csv(pdb_list_file, sep='\t', header=None)
    pdb_list = pdb_list_df[0].tolist()
    seen_pdbs = set()
    filtered_data = []
    for _, row in df.iterrows():
        pdb_id = row['PDB_ID']
        if pdb_id in pdb_list and pdb_id not in seen_pdbs:
            filtered_data.append(row)
            seen_pdbs.add(pdb_id)
    filtered_df = pd.DataFrame(filtered_data)
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered dataset saved to {output_file}")

input_file = "antibody_with_cdr_h123_unique.csv"  
pdb_list_file = "20241107_0591212_summary.tsv"  
output_file = "antibody_with_cdr_h123_unique_with-antigen.csv"

filter_first_unique_pdb(input_file, pdb_list_file, output_file)

