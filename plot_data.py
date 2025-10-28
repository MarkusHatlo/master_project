from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from matplotlib.lines import Line2D


def extract_dims(folder_name: str):
    """
    Pull diameter and height from names like '03_09_D_88mm_350mm'.
    Returns (D_mm, H_mm) as ints or (None, None) if not found.
    """
    mm_nums = re.findall(r'(\d{2,3})\s*mm', folder_name.lower())
    if len(mm_nums) >= 2:
        return int(mm_nums[0]), int(mm_nums[1])
    return None, None

def extract_ER_from_name(name: str):
    # Look for ER or ERp style:
    #   "ER1_0,65" or "ERp_0.65"
    # We'll grab the number part, swap comma->dot, cast to float
    m = re.search(r'ER[p\d]?[ _-]*([0-9]+[.,][0-9]+)', name)
    if not m:
        return np.nan
    val = m.group(1).replace(",", ".")
    try:
        return float(val)
    except ValueError:
        return np.nan
    
def extract_U_from_mat(name: str):
    """
    Pull U from patterns like 'Up_8' or 'Up_20' in the mat_file name.
    Returns float or NaN if not found.
    """
    m = re.search(r'Up_([0-9]+(?:\.[0-9]+)?)', name)
    if not m:
        return np.nan
    try:
        return float(m.group(1))
    except ValueError:
        return np.nan

def marker_for_diameter_local(D):
    # same rules you defined
    if pd.isna(D):
        return "o"
    d = int(D)
    if d == 100: return "s"
    if d == 120: return "^"
    return "o"

def make_height_colors(H_series, nan_color="0.5"):
    """Return (color_for_height_fn, legend_handles)."""
    # unique, sorted known heights
    heights = pd.Series(H_series).dropna().astype(int).sort_values().unique()
    cmap = plt.get_cmap("tab10", max(1, len(heights)))
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

def plot_all(data_avg):
    # --- Plot: mean U vs mean (calculated) ER, one line per folder ---
    plt.figure(figsize=(10, 5))
    for (folder, D, H), g in data_avg.groupby(["folder", "D_mm", "H_mm"]):
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

def plot_split_by_D_simple(data_avg, target_D=88):
    """
    Two subplots: left shows D==target_D, right shows everything else.
    """
    # robust mask (handles NaNs cleanly)
    mask_target = data_avg["D_mm"].eq(target_D)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12, 5), sharex=True, sharey=True
    )

    # --- Left: D == target_D ---
    for (folder, D, H), g in data_avg[mask_target].groupby(["folder", "D_mm", "H_mm"]):
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
    for (folder, D, H), g in data_avg[~mask_target].groupby(["folder", "D_mm", "H_mm"]):
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

def plot_split_by_D(data_avg, target_D=88, show_folder_legends=False):
    """
    Two subplots: left shows D==target_D, right shows everything else.
    Lines sharing the same H_mm use the same color across both subplots.
    """
    mask_target = data_avg["D_mm"].eq(target_D)
    color_for_height, height_handles = make_height_colors(data_avg["H_mm"])

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

    draw_panel(ax_left,  data_avg[mask_target],     f"D = {target_D} mm")
    draw_panel(ax_right, data_avg[~mask_target],    "All other diameters")

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

# Legend proxies for marker shapes (use neutral color so meaning is clear)
marker_handles = [
    Line2D([0], [0], marker="s", linestyle="", color="0.2", markersize=8, label="D = 100 mm"),
    Line2D([0], [0], marker="^", linestyle="", color="0.2", markersize=8, label="D = 120 mm"),
    Line2D([0], [0], marker="o", linestyle="", color="0.2", markersize=8, label="D = 88 mm"),
]


def plot_LBO():
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

    # plot_all()
    plot_split_by_D(avg, 88, True)


def plot_freq_full():
    """
    Make a 3-row figure:
      (1) fft_f0_Hz vs ER
      (2) freq_mean_Hz vs ER
      (3) (fft_f0_Hz - freq_mean_Hz) vs ER

    One line per folder, colored by H_mm, marker by D_mm.
    """

    # --- 1. Load frequency data from logs 4-6 ---
    freq_df = pd.read_csv("freq_results_log456.csv")

    freq_df["ER_guess"] = freq_df["tdms_file"].apply(extract_ER_from_name)

    # --- 2. Extract dimensions from folder name ---
    dims = freq_df["folder"].apply(extract_dims)
    freq_df["D_mm"] = [d for d, h in dims]
    freq_df["H_mm"] = [h for d, h in dims]

    # --- 3. Average repeated runs at same (folder, D, H, ER) ---
    avg_freq = (
        freq_df.groupby(["folder", "D_mm", "H_mm", "ER_guess"], as_index=False)
        .agg(
            fft_f0_Hz    = ("fft_f0_Hz", "mean"),
            freq_mean_Hz = ("freq_mean_Hz", "mean")
        )
    )

    # Compute the difference for subplot 3
    avg_freq["freq_diff_Hz"] = avg_freq["fft_f0_Hz"] - avg_freq["freq_mean_Hz"]

    # --- 4. Styling helpers ---
    color_for_height, height_handles = make_height_colors(avg_freq["H_mm"])

    diameters_present = (
        avg_freq["D_mm"].dropna().astype(int).sort_values().unique()
    )
    marker_handles = []
    for dval in diameters_present:
        mk = marker_for_diameter_local(dval)
        marker_handles.append(
            Line2D(
                [0],[0],
                marker=mk, linestyle="", color="0.2",
                markersize=8, label=f"D = {dval} mm"
            )
        )
    if avg_freq["D_mm"].isna().any():
        marker_handles.append(
            Line2D(
                [0],[0],
                marker="o", linestyle="", color="0.2",
                markersize=8, label="D unknown"
            )
        )

    # --- 5. Figure with 3 subplots ---
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1,
        figsize=(10, 11),
        sharex=True,
        constrained_layout=False
    )

    def draw_panel(ax, ycol, ylabel, title):
        for (folder, D, H), g in avg_freq.groupby(["folder", "D_mm", "H_mm"]):
            g = g.sort_values("ER_guess")

            if pd.notna(D) and pd.notna(H):
                label = f"D={int(D)}mm, H={int(H)}mm"
            else:
                label = folder

            ax.plot(
                g["ER_guess"],
                g[ycol],
                "-",
                marker=marker_for_diameter_local(D),
                markersize=5,
                color=color_for_height(H),
                label=label
            )

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # 1) fft_f0_Hz
    draw_panel(
        ax_top,
        ycol="fft_f0_Hz",
        ylabel="f₀ from FFT [Hz]",
        title="Dominant frequency (FFT peak)"
    )

    # 2) freq_mean_Hz
    draw_panel(
        ax_mid,
        ycol="freq_mean_Hz",
        ylabel="Mean freq [Hz]",
        title="Mean frequency from peak intervals"
    )

    # 3) difference
    draw_panel(
        ax_bot,
        ycol="freq_diff_Hz",
        ylabel="Δf [Hz]",
        title="fft_f0_Hz - freq_mean_Hz"
    )

    # Shared x-label on the bottom plot
    ax_bot.set_xlabel("Equivalence ratio [-]")

    # --- 6. Legends (height colors + diameter markers)
    # We'll pin them under the plots, left/right.
    leg1 = fig.legend(
        handles=height_handles,
        title="Height (color)",
        fontsize=8,
        loc="lower left",
        bbox_to_anchor=(0.08, 0.0)
    )
    fig.legend(
        handles=marker_handles,
        title="Diameter (marker)",
        fontsize=8,
        loc="lower right",
        bbox_to_anchor=(0.92, 0.0)
    )
    # keep leg1 referenced so it doesn't get GC'ed
    _ = leg1

    fig.suptitle("Instability frequency vs Equivalence ratio")
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.95])

    plt.show()

def plot_freq_scatter(csv_path="freq_results_log456.csv"):
    """
    Make a 3-row figure:

      (1) freq_mean_Hz vs U          [U from 'Up_#' in mat_file]
      (2) freq_mean_Hz vs ER         [ER parsed from tdms/mat file]
      (3) freq_mean_Hz vs Volume     [cylinder volume from D_mm,H_mm]

    Steps:
      - read CSV
      - extract D_mm, H_mm from folder
      - extract U from mat_file
      - extract ER from tdms_file / mat_file
      - compute volume_mm3 = pi * (D_mm/2)^2 * H_mm
      - scatter plots
    """

    # --- Load frequency data ---
    freq_df = pd.read_csv(csv_path).copy()

    # --- Dimensions from folder name ---
    dims = freq_df["folder"].apply(extract_dims)
    freq_df["D_mm"] = [d for d, h in dims]
    freq_df["H_mm"] = [h for d, h in dims]

    # --- U from mat_file (Up_### in the .mat filename) ---
    freq_df["U"] = freq_df["mat_file"].astype(str).apply(extract_U_from_mat)

    # --- ER from file names ---
    er_from_tdms = freq_df["tdms_file"].astype(str).apply(extract_ER_from_name)
    er_from_mat  = freq_df["mat_file"].astype(str).apply(extract_ER_from_name)
    freq_df["ER"] = er_from_tdms.fillna(er_from_mat)

    # --- Volume in mm^3 (treat quartz section like a cylinder) ---
    # V = pi * (D/2)^2 * H
    def calc_volume_mm3(row):
        D = row["D_mm"]
        H = row["H_mm"]
        if pd.isna(D) or pd.isna(H):
            return np.nan
        r = float(D) / 2.0  # mm
        return np.pi * (r ** 2) * float(H)  # mm^3

    freq_df["Volume_mm3"] = freq_df.apply(calc_volume_mm3, axis=1)

    # --- Sanity: choose the y column we're plotting ---
    if "fft_f0_Hz" not in freq_df.columns:
        raise ValueError("Expected column 'fft_f0_Hz' in CSV")
    ycol = "fft_f0_Hz"

    # --- Helper to scatter safely ---
    def do_scatter(ax, x_col, x_label):
        sub = freq_df[[x_col, ycol]].dropna()
        ax.scatter(sub[x_col].values, sub[ycol].values)
        ax.set_xlabel(x_label)
        ax.set_ylabel("FFT peak frequency [Hz]")
        ax.grid(True, alpha=0.3)

    # --- Build 3×1 figure ---
    fig, (ax_u, ax_er, ax_vol) = plt.subplots(
        3, 1, figsize=(9, 10), sharex=False
    )

    # 1) freq vs U
    do_scatter(ax_u, "U", "U [m/s]")
    ax_u.set_title("FFT peak frequency vs Speed U")

    # 2) freq vs ER
    do_scatter(ax_er, "ER", "Equivalence ratio [-]")
    ax_er.set_title("FFT peak frequency vs Equivalence ratio")

    # 3) freq vs Volume
    do_scatter(ax_vol, "Volume_mm3", "Volume [mm³]")
    ax_vol.set_title("FFT peak frequency vs Volume (cylindrical quartz)")

    fig.suptitle("Frequency relationships", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()


# plot_freq_full()
plot_freq_scatter()