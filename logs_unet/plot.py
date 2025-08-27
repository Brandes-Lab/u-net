import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv("val_losses.csv")

# # Flip the AUCs because lower scores = more pathogenic
# # df["corrected_auc"] = 1 - df["auc"]

# # Extract training steps and sort
# # df["epoch"] = df["checkpoint"].str.extract(r"checkpoint-(\d+)").astype(int)
# df = df.sort_values("epoch")

# # Plot corrected AUCs
# # plt.plot(df["step"], df["corrected_auc"], marker="o")
# plt.plot(df["epoch"], df["val_loss"], marker="o")
# plt.xlabel("Epoch")
# plt.ylabel("Val Loss")
# plt.title("Val Loss vs Epoch")
# plt.grid(True)
# plt.savefig("/gpfs/data/brandeslab/Project/u-net/logs_unet/val_loss.png")
# plt.show()


df = pd.read_csv("vep_eval.csv")

# Flip the AUCs because lower scores = more pathogenic
# df["corrected_auc"] = 1 - df["auc"]

# Extract training steps and sort
# df["epoch"] = df["checkpoint"].str.extract(r"checkpoint-(\d+)").astype(int)
df = df.sort_values("epoch")

# Plot corrected AUCs
# plt.plot(df["step"], df["corrected_auc"], marker="o")
plt.plot(df["epoch"], df["auc"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Zero shot VEP AUC vs Epoch")
plt.grid(True)
plt.savefig("/gpfs/data/brandeslab/Project/u-net/logs_unet/vep_auc.png")
plt.show()
