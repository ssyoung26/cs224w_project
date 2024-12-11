import sys
import os

# Dynamically add the RefineGNN folder and all its subdirectories to sys.path
refinegnn_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for root, dirs, files in os.walk(refinegnn_dir):
    sys.path.append(root)

from structgen.data_2 import AntibodyDataset2, StructureLoader2, completize

# Initialize dataset and loader
dataset = AntibodyDataset2(jsonl_file="../data/sabdab_2022_01/val_data.jsonl", cdr_type="3", max_len=130)
loader = StructureLoader2(dataset.data, batch_tokens=100)

# Iterate through batches and debug
for batch in loader:
    X_antibody, S_antibody, mask_antibody, X_antigen, S_antigen, mask_antigen = completize(batch)
    print("Antibody coords shape:", X_antibody.shape)
    print("Antigen coords shape:", X_antigen.shape)
    print("Antibody sequence shape:", S_antibody.shape)
    print("Antigen sequence shape:", S_antigen.shape)
    break
