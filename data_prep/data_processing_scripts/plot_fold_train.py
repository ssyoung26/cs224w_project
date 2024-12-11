import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('fold_train_results.csv')

df['Epoch'] = df['Epoch'].fillna(method='ffill')

plt.figure(figsize=(10, 5))
plt.plot(df.groupby('Epoch')['RMSD'].max(), marker='o', linestyle='-', color='b')
plt.xlabel('Epoch')
plt.ylabel('RMSD')
plt.title('RMSD vs. Epoch')
plt.grid(True)
plt.savefig('plot_rmsd_vs_epoch.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Train'], marker='o', linestyle='-', color='r')
plt.xlabel('Step')
plt.ylabel('Train Loss')
plt.title('Train Loss per Step')
plt.grid(True)
plt.savefig('plot_train_loss_per_step.png')
plt.close()
