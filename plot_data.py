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

from matplotlib.lines import Line2D

def make_height_colors(H_series, nan_color="0.5"):
    """Return (color_for_height_fn, legend_handles)."""
    # unique, sorted known heights
    heights = pd.Series(H_series).dropna().astype(int).sort_values().unique()
    cmap = plt.cm.get_cmap("tab10", max(1, len(heights)))
    color_map = {h: cmap(i) for i, h in enumerate(heights)}

    def color_for_height(H):
        if pd.isna(H):
            return nan_color
        return color_map.get(int(H), nan_color)

    # legend proxies for heights
    handles = [Line2D([0], [0], lw=3, color=color_map[h], label=f"H={h} mm") for h in heights]
    if pd.isna(H_series).any():
        handles.append(Line2D([0], [0], lw=3, color=nan_color, label="H unknown"))
    return color_for_height, handles

from matplotlib.lines import Line2D

def marker_for_diameter(D):
    if pd.isna(D): 
        return "o"
    d = int(D)
    if d == 100: return "s"   # square
    if d == 120: return "^"   # triangle
    return "o"                # default

# Legend proxies for marker shapes (use neutral color so meaning is clear)
marker_handles = [
    Line2D([0], [0], marker="s", linestyle="", color="0.2", markersize=8, label="D = 100 mm"),
    Line2D([0], [0], marker="^", linestyle="", color="0.2", markersize=8, label="D = 120 mm"),
    Line2D([0], [0], marker="o", linestyle="", color="0.2", markersize=8, label="D = 88 mm"),
]


# --- Load ---
df = pd.read_csv("post_process_data.csv")

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

def plot_all():
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

def plot_split_by_D_simple(target_D=88):
    """
    Two subplots: left shows D==target_D, right shows everything else.
    """
    # robust mask (handles NaNs cleanly)
    mask_target = avg["D_mm"].eq(target_D)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12, 5), sharex=True, sharey=True
    )

    # --- Left: D == target_D ---
    for (folder, D, H), g in avg[mask_target].groupby(["folder", "D_mm", "H_mm"]):
        g = g.sort_values("ER_mean")
        label = (f"D={D}mm, H={H}mm"
                 if pd.notna(D) and pd.notna(H) else folder)
        ax_left.plot(g["ER_mean"], g["U_mean"], "o-", label=label)

    ax_left.set_title(f"D = {target_D} mm")
    ax_left.set_xlabel("Equivalence ratio [-]")
    ax_left.set_ylabel("Mean velocity [m/s]")
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(title="Quartz dimensions", fontsize=9)

    # --- Right: all other D ---
    for (folder, D, H), g in avg[~mask_target].groupby(["folder", "D_mm", "H_mm"]):
        g = g.sort_values("ER_mean")
        label = (f"D={D}mm, H={H}mm"
                 if pd.notna(D) and pd.notna(H) else folder)
        ax_right.plot(g["ER_mean"], g["U_mean"], "o-", label=label)

    ax_right.set_title("All other diameters")
    ax_right.set_xlabel("Equivalence ratio [-]")
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(title="Quartz dimensions", fontsize=9)

    fig.suptitle("Mean velocity vs Equivalence ratio")
    fig.tight_layout()
    plt.show()

def plot_split_by_D(target_D=88, show_folder_legends=False):
    """
    Two subplots: left shows D==target_D, right shows everything else.
    Lines sharing the same H_mm use the same color across both subplots.
    """
    mask_target = avg["D_mm"].eq(target_D)
    color_for_height, height_handles = make_height_colors(avg["H_mm"])

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # helper to draw one panel
    def draw_panel(ax, data, title):
        for (folder, D, H), g in data.groupby(["folder", "D_mm", "H_mm"]):
            g = g.sort_values("ER_mean")
            label = (f"D={D}mm, H={H}mm" if pd.notna(D) and pd.notna(H) else folder)
            ax.plot(
                g["ER_mean"], g["U_mean"],
                "-",                           # line
                marker=marker_for_diameter(D), # shape by D
                markersize=5,
                color=color_for_height(H),     # color by H
                label=label
            )
        ax.set_title(title)
        ax.set_xlabel("Equivalence ratio [-]")
        ax.set_ylabel("Mean velocity [m/s]")
        ax.grid(True, alpha=0.3)
        if show_folder_legends:
            ax.legend(title="Quartz dimensions", fontsize=8)

    draw_panel(ax_left,  avg[mask_target],     f"D = {target_D} mm")
    draw_panel(ax_right, avg[~mask_target],    "All other diameters")

    # existing height legend (colors)
    fig.legend(handles=height_handles, title="Height (color)",
            loc="lower left", ncol=min(5, len(height_handles)),
            bbox_to_anchor=(0.5, 0.05))

    # new marker legend (shapes)
    fig.legend(handles=marker_handles, title="Diameter (marker)",
            loc="lower right", bbox_to_anchor=(0.5, 0.05))

    fig.suptitle("Mean velocity vs Equivalence ratio")
    fig.tight_layout(rect=[0.02, 0.18, 0.98, 0.92])  # [left, bottom, right, top]  
    plt.show()


# plot_all()
plot_split_by_D(88,True)