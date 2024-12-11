import pandas as pd
import matplotlib.pyplot as plt

file_data_val = "data_val.csv"
file_diffusion_val = "diffusion_val.csv"
file_test_ppl_rmsd = "test_ppl_rmsd.csv"
file_train_curves = "train_curves.csv"

df_data_val = pd.read_csv(file_data_val)
df_diffusion_val = pd.read_csv(file_diffusion_val)
df_test_ppl_rmsd = pd.read_csv(file_test_ppl_rmsd)
df_train_curves = pd.read_csv(file_train_curves)

def save_plots():
    # Plot 1: Training Curves - Line plot of Train PPL vs Batch Number
    plt.figure(figsize=(10, 6))
    for col in df_train_curves.columns[1:]:
        plt.plot(df_train_curves["Batch number"], df_train_curves[col], label=col)
    plt.title("Training Curves - PPL vs Batch Number")
    plt.xlabel("Batch Number")
    plt.ylabel("Train PPL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("train_curves_plot.png")
    plt.close()

    # Plot 2: Validation Curves - Line plot of Validation PPL and RMSD vs Epoch
    df_data_val_clean = df_data_val.iloc[1:].reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    for col, rmsd_col, label in zip(df_data_val.columns[1:7], df_data_val.columns[7:], df_data_val.columns[1:7]):
        plt.plot(df_data_val_clean["Unnamed: 0"], pd.to_numeric(df_data_val_clean[col]), label=f"{label} PPL", marker='o')
        plt.plot(df_data_val_clean["Unnamed: 0"], pd.to_numeric(df_data_val_clean[rmsd_col]), linestyle="--", label=f"{label} RMSD", marker='x')
    plt.title("Validation Curves - PPL and RMSD vs Epoch")
    plt.xlim(0, 12)
    plt.xlabel("Epoch")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data_val_plot.png")
    plt.close()

    # Plot 3: Diffusion Validation - Line plot of Validation PPL and RMSD vs Epoch
    df_diffusion_val_clean = df_diffusion_val.iloc[1:].reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    for col, rmsd_col, label in zip(df_diffusion_val.columns[1:7], df_diffusion_val.columns[7:], df_diffusion_val.columns[1:7]):
        plt.plot(df_diffusion_val_clean["Unnamed: 0"], pd.to_numeric(df_diffusion_val_clean[col]), label=f"{label} PPL", marker='o')
        plt.plot(df_diffusion_val_clean["Unnamed: 0"], pd.to_numeric(df_diffusion_val_clean[rmsd_col]), linestyle="--", label=f"{label} RMSD", marker='x')
    plt.title("Diffusion Validation - PPL and RMSD vs Epoch")
    plt.xlim(0, 12)
    plt.xlabel("Epoch")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("diffusion_val_plot.png")
    plt.close()

    # Plot 4: Test PPL and RMSD - Bar chart
    plt.figure(figsize=(10, 6))
    x = df_test_ppl_rmsd["Unnamed: 0"]
    ppl_values = df_test_ppl_rmsd["Test PPL"]
    rmsd_values = df_test_ppl_rmsd["Test RMSD"]
    bar_width = 0.35
    plt.bar(x, ppl_values, width=bar_width, label="Test PPL")
    plt.bar(x, rmsd_values, width=bar_width, label="Test RMSD", alpha=0.7)
    plt.title("Test PPL and RMSD")
    plt.xlabel("Model")
    plt.ylabel("Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig("test_ppl_rmsd_bar_chart.png")
    plt.close()
save_plots()

"Plots have been generated and saved successfully."

