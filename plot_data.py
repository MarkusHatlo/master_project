from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


def extract_dims(folder_name: str):
    """
    Pull diameter and height from names like '03_09_D_88mm_350mm'.
    Returns (D_mm, H_mm) as ints or (None, None) if not found.
    """
    mm_nums = re.findall(r'(\d{2,3})\s*mm', folder_name.lower())
    if len(mm_nums) >= 2:
        return int(mm_nums[0]), int(mm_nums[1])
    return None, None

# --- Load ---
df = pd.read_csv("post_process_data_all.csv")

# Keep only Log 1-3 and rows with an estimated ER to group logs together
df = df[df["log"].isin([1, 2, 3])].dropna(subset=["er_est"])

# --- Average within each folder & estimated ER ---
# We take the MEAN of the *calculated* ER and U across logs 1-3
avg = (df.groupby(["folder", "er_est"], as_index=False)
         .agg(ER_mean=("ER", "mean"),
              U_mean=("velocity", "mean")))

# Optional: nicer legend labels with D/H if present in folder name
dims = avg["folder"].apply(extract_dims)
avg["D_mm"] = [d for d, h in dims]
avg["H_mm"] = [h for d, h in dims]

# --- Plot: mean U vs mean (calculated) ER, one line per folder ---
plt.figure(figsize=(10, 5))
for (folder, D, H), g in avg.groupby(["folder", "D_mm", "H_mm"]):
    g = g.sort_values("ER_mean")
    label = (f"D={D}mm, H={H}mm"
             if pd.notna(D) and pd.notna(H) else folder)
    plt.plot(g["ER_mean"],g["U_mean"], "o-", label=label)

plt.xlabel("Equivalence ratio [-]")
plt.ylabel("Mean velocity [m/s]")
plt.title("Mean velocity vs Equivalence ratio")
plt.grid(True, alpha=0.3)
plt.legend(title="Quartz dimentions", fontsize=9)
plt.tight_layout()
plt.show()
