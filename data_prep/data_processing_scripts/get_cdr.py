import pandas as pd

def extract_fab_sequences(input_file, output_file):
    """
    Process the CSV to extract H1, H2, and H3 sequences for each unique PDB entry, with improved parsing.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
    """
    df = pd.read_csv(input_file)
    processed_data = []
    for index, row in df.iterrows():
        try:
            pdb_id, species, method, resolution, cell_data = row
        except ValueError:
            print(f"Skipping row {index} due to incorrect formatting: {row}")
            continue 

        print(f"Processing PDB ID: {pdb_id}")
        fab_entries = cell_data.split("Fab")  
        print(f"Fab Entries Found: {fab_entries}")

        for entry in fab_entries:
            entry = entry.strip()
            if not entry: 
                continue

            parts = entry.split(":", 1)
            if len(parts) < 2:
                print(f"Skipping malformed Fab entry: {entry}")
                continue  

            fab_name = parts[0].strip()
            raw_sections = parts[1].strip()
            sections = raw_sections.split("\n")
            print(f"Fab Name: {fab_name}")
            print(f"Sections: {sections}")

            # Initialize H1, H2, H3 values
            h1 = h2 = h3 = ""
            for section in sections:
                section = section.strip()
                if section.startswith("H1:"):
                    h1 = section.split(":", 1)[1].strip()
                    print(f"Extracted H1: {h1}")
                elif section.startswith("H2:"):
                    h2 = section.split(":", 1)[1].strip()
                    print(f"Extracted H2: {h2}")
                elif section.startswith("H3:"):
                    h3 = section.split(":", 1)[1].strip()
                    print(f"Extracted H3: {h3}")

            # Store the processed data
            processed_data.append({
                "PDB_ID": pdb_id,
                "Species": species,
                "Method": method,
                "Resolution": resolution,
                "Fab": fab_name,
                "H1": h1,
                "H2": h2,
                "H3": h3
            })

    result_df = pd.DataFrame(processed_data)
    result_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

input_file = "antibody_with_cdr.csv"  
output_file = "antibody_with_cdr_h123.csv"
extract_fab_sequences(input_file, output_file)

