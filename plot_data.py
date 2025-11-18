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

    # oscillations_U_88_1 = [7,8,14,20,26,8,15,39]
    # oscillations_ER_88_1 = [0.6,0.65,0.7,0.71,0.72,0.6,0.7,0.75]
    # oscillations_U_88_2 = [28,35,42,53,50,60]
    # oscillations_ER_88_2 = [0.72,0.75,0.8,0.85,0.8,0.85]
    # oscillations_U_2 = [45,53,20,32,42,51]
    # oscillations_ER_2 = [0.90,0.95,0.75,0.8,0.85,0.9]
    # oscillations_U_3 = [12,17,26,8,14]
    # oscillations_ER_3 = [0.7,0.75,0.8,0.66,0.7]

    #Strict
    oscillations_U_88_1 = [7,8,14,20,26,8,15,39]
    oscillations_ER_88_1 = [0.6,0.65,0.7,0.71,0.72,0.6,0.7,0.75]
    oscillations_U_88_2 = [14,42,50]
    oscillations_ER_88_2 = [0.7,0.8,0.8]
    oscillations_U_88_3 = [60,28,35,53]
    oscillations_ER_88_3 = [0.85,0.72,0.75,0.85]
    
    oscillations_U_2 = [45,53,32,42]
    oscillations_ER_2 = [0.90,0.95,0.8,0.85]
    oscillations_U_3 = [12,17,26,8,14,20,51]
    oscillations_ER_3 = [0.7,0.75,0.8,0.66,0.7,0.75,0.9]

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
    ax_left.plot(oscillations_ER_88_1,oscillations_U_88_1,marker='*', linestyle='',color='green')
    ax_left.plot(oscillations_ER_88_2,oscillations_U_88_2,marker='*', linestyle='',color='orange')
    ax_left.plot(oscillations_ER_88_3,oscillations_U_88_3,marker='*', linestyle='',color='red')
    ax_right.plot(oscillations_ER_2,oscillations_U_2,marker='*', linestyle='',color='orange')
    ax_right.plot(oscillations_ER_3,oscillations_U_3,marker='*', linestyle='',color='red')
    # ax_right.set_xlim(0,1)
    # ax_left.set_xlim(0,1)


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


def plot_LBO(csv_name: str):
    # --- Load ---
    df = pd.read_csv(csv_name)

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

def plot_freq_full(csv_name: str):
    """
    Make a 3-row figure:
      (1) fft_f0_Hz vs ER
      (2) freq_mean_Hz vs ER
      (3) (fft_f0_Hz - freq_mean_Hz) vs ER

    One line per folder, colored by H_mm, marker by D_mm.
    """

    # --- 1. Load frequency data from ---
    freq_df = pd.read_csv(csv_name)

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

def load_or_make_avg_freq(
    src_csv: str = "freq_results.csv",
    avg_csv: str = "freq_avg_by_folder_D_H_ER.csv",
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Return a tidy dataframe averaged over (folder, D_mm, H_mm, ER_guess).
    If avg_csv exists and force_recompute=False, load it; else compute and save.
    """
    avg_path = Path(avg_csv)
    if avg_path.exists() and not force_recompute:
        return pd.read_csv(avg_path)

    # Compute fresh
    df = pd.read_csv(src_csv)

    # Expect these helpers to exist in your codebase
    df["ER_guess"] = df["tdms_file"].apply(extract_ER_from_name)

    dims = df["folder"].apply(extract_dims)  # returns (D_mm, H_mm)
    df["D_mm"] = [d for d, h in dims]
    df["H_mm"] = [h for d, h in dims]

    required_cols = {"folder","D_mm","H_mm","ER_guess","fft_f0_Hz","freq_mean_Hz"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in source CSV: {missing}")

    avg_freq = (
        df.groupby(["folder", "D_mm", "H_mm", "ER_guess"], as_index=False)
          .agg(
              fft_f0_Hz    = ("fft_f0_Hz", "mean"),
              freq_mean_Hz = ("freq_mean_Hz", "mean"),
          )
          .sort_values(["folder","D_mm","H_mm","ER_guess"])
          .reset_index(drop=True)
    )
    avg_freq["freq_diff_Hz"] = avg_freq["fft_f0_Hz"] - avg_freq["freq_mean_Hz"]

    # Save cache for next time
    avg_freq.to_csv(avg_path, index=False)
    return avg_freq

def plot_freq_mean_vs_f0(
    avg_csv: str = "freq_avg_by_folder_D_H_ER.csv",
    recompute: bool = False,
    src_csv: str = "freq_results_8_windows_with_peaks.csv",
):
    """
    Scatter/line plot comparing freq_mean_Hz (y) vs fft_f0_Hz (x),
    using pre-averaged data if available.
    - Set recompute=True to rebuild avg CSV from src_csv.
    """
    if recompute or not Path(avg_csv).exists():
        avg_freq = load_or_make_avg_freq(src_csv=src_csv, avg_csv=avg_csv, force_recompute=recompute)
    else:
        avg_freq = pd.read_csv(avg_csv)

    if avg_freq.empty:
        print("No averaged data to plot.")
        return

    color_for_height, height_handles = make_height_colors(avg_freq["H_mm"])

    diameters_present = avg_freq["D_mm"].dropna().astype(int).sort_values().unique()
    marker_handles = []
    for dval in diameters_present:
        mk = marker_for_diameter_local(dval)
        marker_handles.append(
            Line2D([0],[0], marker=mk, linestyle="", color="0.2",
                   markersize=8, label=f"D = {dval} mm")
        )
    if avg_freq["D_mm"].isna().any():
        marker_handles.append(
            Line2D([0],[0], marker="o", linestyle="", color="0.2",
                   markersize=8, label="D unknown")
        )

    fig, ax = plt.subplots(figsize=(7, 7))

    # One series per (folder, D, H); connect points by ER for visual continuity
    for (folder, D, H), g in avg_freq.groupby(["folder", "D_mm", "H_mm"]):
        g = g.sort_values("ER_guess")
        label = f"{folder}" if (pd.isna(D) or pd.isna(H)) else f"D={int(D)}mm, H={int(H)}mm"

        ax.plot(
            g["fft_f0_Hz"],            # x
            g["freq_mean_Hz"],         # y
            marker=marker_for_diameter_local(D),
            linestyle="",
            markersize=5,
            linewidth=1.0,
            color=color_for_height(H),
            label=label
        )

    # y = x reference
    xvals = avg_freq["fft_f0_Hz"].to_numpy()
    yvals = avg_freq["freq_mean_Hz"].to_numpy()
    lo = np.nanmin([xvals.min(), yvals.min()])
    hi = np.nanmax([xvals.max(), yvals.max()])
    if np.isfinite(lo) and np.isfinite(hi):
        pad = 0.02 * (hi - lo if hi > lo else 1.0)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], "--", linewidth=1.0, color="0.5")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel("FFT frequency [Hz]")
    ax.set_ylabel("Counted peaks frequency [Hz]")
    ax.set_title("Counted peaks frequency vs FFT frequency")
    ax.grid(True, alpha=0.3)

    # Legends
    leg1 = fig.legend(
        handles=height_handles,
        title="Height (color)",
        fontsize=8,
        loc="lower left",
        bbox_to_anchor=(0.05, 0.0)
    )
    fig.legend(
        handles=marker_handles,
        title="Diameter (marker)",
        fontsize=8,
        loc="lower right",
        bbox_to_anchor=(0.95, 0.0)
    )
    _ = leg1

    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.98])
    plt.show()

def plot_freq_points_vs_f0(
    src_csv="freq_results_log456_last.csv",
    point_alpha=0.85,
    marker_size_pts=6,
):
    df = pd.read_csv(src_csv)

    # If you’re deriving D/H from folder names:
    dims = df["folder"].apply(extract_dims)  # must return (D_mm, H_mm)
    df["D_mm"] = [d for d, h in dims]
    df["H_mm"] = [h for d, h in dims]

    # Color by height, just like before
    color_for_height, height_handles = make_height_colors(df["H_mm"])

    # Marker by diameter, just like before
    diameters_present = (
        df["D_mm"].dropna().astype(int).sort_values().unique()
        if "D_mm" in df.columns else []
    )
    marker_handles = []
    for dval in diameters_present:
        mk = marker_for_diameter_local(dval)
        marker_handles.append(
            Line2D([0],[0], marker=mk, linestyle="", color="0.2",
                   markersize=8, label=f"D = {dval} mm")
        )
    if df["D_mm"].isna().any():
        marker_handles.append(
            Line2D([0],[0], marker="o", linestyle="", color="0.2",
                   markersize=8, label="D unknown")
        )

    fig, ax = plt.subplots(figsize=(7, 7))

    # Batch by (D,H) so each subgroup gets its marker+color
    for (D, H), g in df.groupby(["D_mm", "H_mm"]):
        mk = marker_for_diameter_local(int(D)) if pd.notna(D) else "o"
        c  = color_for_height(H)

        ax.errorbar(
            g["fft_f0_Hz"].to_numpy(),
            g["freq_mean_Hz"].to_numpy(),
            yerr=g["freq_std_Hz"].to_numpy(),
            fmt=mk,                 # marker shape from diameter
            linestyle="",           # points only
            ms=marker_size_pts,     # marker size in points (errorbar uses 'ms')
            elinewidth=1.0,
            capsize=2.5,
            alpha=point_alpha,
            color=c,                # color from height
        )

    # y = x and symmetric-ish limits
    xvals = df["fft_f0_Hz"].to_numpy()
    yvals = df["freq_mean_Hz"].to_numpy()
    lo = np.nanmin([np.nanmin(xvals), np.nanmin(yvals)])
    hi = np.nanmax([np.nanmax(xvals), np.nanmax(yvals)])
    pad = 0.02 * (hi - lo if hi > lo else 1.0)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1.0, color="0.5")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("FFT frequency [Hz]")
    ax.set_ylabel("Counted peaks frequency [Hz]")
    ax.set_title("Counted peaks vs FFT — all points")
    ax.grid(True, alpha=0.3)

    # Legends (same layout philosophy as before)
    leg1 = fig.legend(
        handles=height_handles,
        title="Height (color)",
        fontsize=8,
        loc="lower left",
        bbox_to_anchor=(0.05, 0.0)
    )
    fig.legend(
        handles=marker_handles,
        title="Diameter (marker)",
        fontsize=8,
        loc="lower right",
        bbox_to_anchor=(0.95, 0.0)
    )
    _ = leg1

    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.98])
    plt.show()
    
def plot_freq_f0_and_a0(csv_path):
    """
    Two-panel figure for logs 4–6 style CSVs:
      (1) fft_f0_Hz vs ER
      (2) fft_a0_amp vs ER

    Lines: one per folder, color = H_mm, marker = D_mm.
    """
    # --- Load ---
    df = pd.read_csv(csv_path)
    required = {"folder", "mat_file", "tdms_file", "log_no", "fft_f0_Hz", "fft_a0_amp"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    # --- Parse ER, D, H ---
    df["ER_guess"] = df["tdms_file"].astype(str).apply(extract_ER_from_name)
    dims = df["folder"].astype(str).apply(extract_dims)
    df["D_mm"] = [d for d, h in dims]
    df["H_mm"] = [h for d, h in dims]

    # drop rows without ER (nothing to plot on x)
    df = df.dropna(subset=["ER_guess"])

    # --- Average repeats at same (folder, D, H, ER) ---
    avg = (df.groupby(["folder", "D_mm", "H_mm", "ER_guess"], as_index=False)
             .agg(fft_f0_Hz=("fft_f0_Hz", "mean"),
                  fft_a0_amp=("fft_a0_amp", "mean")))

    if avg.empty:
        raise ValueError("No rows left after parsing ER. Check file names for ER patterns (e.g. 'ERp_0.75').")

    # --- Styling helpers ---
    color_for_height, height_handles = make_height_colors(avg["H_mm"])
    diameters_present = avg["D_mm"].dropna().astype(int).sort_values().unique()
    marker_handles = [Line2D([0],[0], marker=marker_for_diameter_local(d),
                             linestyle="", color="0.2", markersize=8, label=f"D = {d} mm")
                      for d in diameters_present]
    if avg["D_mm"].isna().any():
        marker_handles.append(Line2D([0],[0], marker="o", linestyle="", color="0.2",
                                     markersize=8, label="D unknown"))

    # --- Figure ---
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=False)

    def draw_panel(ax, ycol, ylabel, title):
        for (folder, D, H), g in avg.groupby(["folder", "D_mm", "H_mm"]):
            g = g.sort_values("ER_guess")
            label = (f"D={int(D)}mm, H={int(H)}mm") if pd.notna(D) and pd.notna(H) else folder
            ax.plot(g["ER_guess"], g[ycol], "-",
                    marker=marker_for_diameter_local(D), markersize=5,
                    color=color_for_height(H), label=label)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    draw_panel(ax_top, "fft_f0_Hz",  "f₀ (FFT peak) [Hz]",  "Dominant frequency vs ER")
    draw_panel(ax_bot, "fft_a0_amp", "a₀ amplitude [a.u.]", "Peak amplitude vs ER")
    ax_bot.set_xlabel("Equivalence ratio [-]")
    ax_bot.axhline(y=0.07,color='red')

    # Legends
    leg1 = fig.legend(handles=height_handles, title="Height (color)", fontsize=8,
                      loc="lower left", bbox_to_anchor=(0.08, 0.0))
    fig.legend(handles=marker_handles, title="Diameter (marker)", fontsize=8,
               loc="lower right", bbox_to_anchor=(0.92, 0.0))
    _ = leg1

    fig.suptitle("Instability metrics vs Equivalence ratio")
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.95])
    plt.show()

def plot_freq_scatter(csv_path):
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
    # freq_df["U"] = freq_df['U_blow_off_m_per_s']

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

    do_manual = False
    LBO_manual = False
    # --- Build 3×1 figure ---
    fig, (ax_u) = plt.subplots(
        1, 1, figsize=(8, 8), sharex=False
    )

    # 1) freq vs U
    if do_manual:
        # log 1 2 3
        if LBO_manual:
            ax_u.set_xlabel('U [m/s]')
            ax_u.set_ylabel("FFT peak frequency [Hz]")
            ax_u.grid(True, alpha=0.3)
            manual_freq_88_200 = [1.2,0.65,0.6,0.5,0.47,0.58]
            manual_vel_88_200 = [9,15.6,28.6,35.7,43,54]
            ax_u.plot(manual_vel_88_200,manual_freq_88_200,color='blue',linestyle='', marker='o', label='D: 88 H: 200')

            manual_freq_88_260 = [1.2,1.3,3,4.5,5.2]
            manual_vel_88_260 = [16,41,51,60,67]
            ax_u.plot(manual_vel_88_260,manual_freq_88_260,color='purple',linestyle='',marker='o', label='D: 88 H: 260')

            manual_freq_88_350 = [0.82,1.9,1.72,2.2]
            manual_vel_88_350 = [9,21,16,27]
            ax_u.plot(manual_vel_88_350,manual_freq_88_350,color='green',linestyle='',marker='o', label='D: 88 H: 350')

            manual_freq_100_260 = [0.37,0.2,0.23]
            manual_vel_100_260 = [23,46,54]
            ax_u.plot(manual_vel_100_260,manual_freq_100_260,color='red',linestyle='',marker='o', label='D: 100 H: 260')
        else:
            # log 4 5 6
            #88 200,88 260,88 350,100 260,120 260        
            ax_u.set_xlabel('U [m/s]')
            ax_u.set_ylabel("FFT peak frequency [Hz]")
            ax_u.grid(True, alpha=0.3)
            manual_freq_88_200 = [0.59,0.87,1.17,1.41]
            manual_vel_88_200 = [14,28,35,42]
            ax_u.plot(manual_vel_88_200,manual_freq_88_200,color='blue',linestyle='', marker='o', label='D: 88 H: 200')

            manual_freq_88_260 = [0.59,1.19,2.19,1.99,1.89]
            manual_vel_88_260 = [8,15,39.4,49.7,59.6]
            ax_u.plot(manual_vel_88_260,manual_freq_88_260,color='purple',linestyle='',marker='o', label='D: 88 H: 260')

            manual_freq_88_350 = [0.72,0.62,1.63,2.34,2.21]
            manual_vel_88_350 = [7,8,14,20,26]
            ax_u.plot(manual_vel_88_350,manual_freq_88_350,color='green',linestyle='',marker='o', label='D: 88 H: 350')

            manual_freq_100_260 = [1.17,0.72,0.56,1.04,1.14]
            manual_vel_100_260 = [14,20,22.44,32,42]
            ax_u.plot(manual_vel_100_260,manual_freq_100_260,color='red',linestyle='',marker='o', label='D: 100 H: 260')

        # manual_freq_120_260 = [1.15,0.85,]
        # manual_vel_120_260 = [7,17,26]
        # ax_u.plot(manual_vel_120_260,manual_freq_120_260,color='orange',linestyle='',marker='o', label='D: 100 H: 260')

    else:
        do_scatter(ax_u, "U", "U [m/s]")

    SuTong_freq = [0.5,1,2,4.25,7.75,9]
    SuTong_vel = [10,20,35,60,85,110]
    ax_u.plot(SuTong_vel,SuTong_freq,color='black',linestyle='',marker='o', label='Su Tong')

    ax_u.set_title("FFT peak frequency vs Speed U")
    ax_u.legend()


    # # 2) freq vs ER
    # if do_manual:
    #     ax_er.set_xlabel('Equivalence ratio [-]')
    #     ax_er.set_ylabel("FFT peak frequency [Hz]")
    #     ax_er.grid(True, alpha=0.3)
    #     manual_er_88_200 = [0.7,0.72,0.75,0.8]
    #     ax_er.plot(manual_er_88_200,manual_freq_88_200,color='blue',linestyle='', marker='o', label='D: 88 H: 200')

    #     manual_er_88_260 = [0.6,0.7,0.75,0.8,0.85]
    #     ax_er.plot(manual_er_88_260,manual_freq_88_260,color='purple',linestyle='',marker='o', label='D: 88 H: 260')

    #     manual_er_88_350 = [0.6,0.65,0.70,0.71,0.72]
    #     ax_er.plot(manual_er_88_350,manual_freq_88_350,color='green',linestyle='',marker='o', label='D: 88 H: 350')

    #     manual_er_100_260 = [0.7,0.75,0.73,0.79,0.84]
    #     ax_er.plot(manual_er_100_260,manual_freq_100_260,color='red',linestyle='',marker='o', label='D: 100 H: 260')
    # else:
    #     do_scatter(ax_er, "ER", "Equivalence ratio [-]")

    # ax_er.set_title("FFT peak frequency vs Equivalence ratio")
    # ax_er.legend()

    # # 3) freq vs Volume
    # if do_manual:
    #     ax_vol.set_xlabel('Volume [mm³]')
    #     ax_vol.set_ylabel("FFT peak frequency [Hz]")
    #     ax_vol.grid(True, alpha=0.3)
    #     manual_vol_88_200 = [1.21,1.21,1.21,1.21]
    #     ax_vol.plot(manual_vol_88_200,manual_freq_88_200,color='blue',linestyle='', marker='o', label='D: 88 H: 200')

    #     manual_vol_88_260 = [1.58,1.58,1.58,1.58,1.58]
    #     ax_vol.plot(manual_vol_88_260,manual_freq_88_260,color='purple',linestyle='',marker='o', label='D: 88 H: 260')

    #     manual_vol_88_350 = [2.13,2.13,2.13,2.13,2.13]
    #     ax_vol.plot(manual_vol_88_350,manual_freq_88_350,color='green',linestyle='',marker='o', label='D: 88 H: 350')

    #     manual_vol_100_260 = [2.04,2.04,2.04,2.04,2.04]
    #     ax_vol.plot(manual_vol_100_260,manual_freq_100_260,color='red',linestyle='',marker='o', label='D: 100 H: 260')
    # else:
    #     do_scatter(ax_vol, "Volume_mm3", "Volume [mm³]")
    
    # ax_vol.set_title("FFT peak frequency vs Volume (cylindrical quartz)")
    # ax_vol.legend()


    # fig.suptitle("Frequency relationships", fontsize=14)
    # fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()


def plot_freq_vs_U_by_geom5(csv_path):
    """
    Plot FFT peak frequency vs U, with separate lines for each (D_mm, H_mm).
    Colors encode H_mm, markers encode D_mm.
    """
    # --- Load frequency data ---
    freq_df = pd.read_csv(csv_path).copy()

    # --- Dimensions from folder name ---
    dims = freq_df["folder"].apply(extract_dims)  # returns (D_mm, H_mm)
    freq_df["D_mm"] = [d for d, h in dims]
    freq_df["H_mm"] = [h for d, h in dims]

    # --- U from mat_file (Up_### in the .mat filename) ---
    freq_df["U"] = freq_df["mat_file"].astype(str).apply(extract_U_from_mat)

    # --- Only rows with both U and ycol defined ---
    freq_df = freq_df.dropna(subset=["U", 'fft_f0_Hz'])

    # === Encoding: color for height, marker for diameter ===
    heights = np.sort(freq_df["H_mm"].unique())
    diams   = np.sort(freq_df["D_mm"].unique())

    # Colors by height
    cmap = plt.get_cmap("tab10" if len(heights) <= 10 else "tab20")
    color_for_height = {
        h: cmap(i % cmap.N) for i, h in enumerate(heights)
    }

    # Markers by diameter
    marker_list = ["o", "s", "D", "^", "v", "P", "X", "*", ">", "<"]
    marker_for_diam = {
        d: marker_list[i % len(marker_list)] for i, d in enumerate(diams)
    }

    fig, ax = plt.subplots(figsize=(8, 8))

    # --- Plot data: one line per (D, H) ---
    for (D, H), sub in freq_df.groupby(["D_mm", "H_mm"]):
        sub = sub.sort_values("U")
        ax.plot(
            sub["U"].values,
            sub['fft_f0_Hz'].values,
            linestyle="-",
            marker=marker_for_diam[D],
            color=color_for_height[H],
            alpha=0.9,
            label=f"D = {D} mm, H = {H} mm",
        )

    # --- Su Tong reference curve ---
    SuTong_freq = [0.5, 1, 2, 4.25, 7.75, 9]
    SuTong_vel  = [10, 20, 35, 60, 85, 110]
    ax.plot(
        SuTong_vel,
        SuTong_freq,
        color="black",
        linestyle="-",
        marker="o",
        label="Su Tong"
    )

    # --- Legends in the "separated" style (color = H, marker = D) ---
    height_handles = [
        Line2D([0], [0], color=color_for_height[h], lw=3, label=f"H = {h} mm")
        for h in heights
    ]
    marker_handles = [
        Line2D([0], [0], marker=marker_for_diam[d], linestyle="", color="0.2",
               markersize=8, label=f"D = {d} mm")
        for d in diams
    ]

    leg1 = ax.legend(
        handles=height_handles,
        title="Height",
        loc="upper left",
        frameon=True
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=marker_handles + [Line2D([0],[0], color="black", marker="o",
                                         label="Su Tong")],
        title="Diameter & Ref.",
        loc="lower right",
        frameon=True
    )

    # --- Axes formatting ---
    ax.set_xlabel("U [m/s]")
    ax.set_ylabel("FFT peak frequency [Hz]")
    ax.set_title("FFT peak frequency vs Speed U (separated by geometry)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_freq_vs_U_by_geom(csv_path):
    """
    Plot FFT peak frequency vs U, with lines separated by (D_mm, H_mm),
    using the same color + marker system as before:
      - color_for_height from make_height_colors
      - markers from marker_for_diameter_local
    """
    # --- Load frequency data ---
    freq_df = pd.read_csv(csv_path).copy()

    # --- Dimensions from folder name ---
    dims = freq_df["folder"].apply(extract_dims)  # returns (D_mm, H_mm)
    freq_df["D_mm"] = [d for d, h in dims]
    freq_df["H_mm"] = [h for d, h in dims]

    # --- U from mat_file (Up_### in the .mat filename) ---
    freq_df["U"] = freq_df["mat_file"].astype(str).apply(extract_U_from_mat)

    # --- Keep only rows with both U and frequency defined ---
    freq_df = freq_df.dropna(subset=["U", 'fft_f0_Hz'])

    # --- Colors: same system as before ---
    # This gives you: dict H_mm -> color, and legend handles for heights
    color_for_height, height_handles = make_height_colors(freq_df["H_mm"])

    # --- Markers: same system as before ---
    diameters_present = (
        freq_df["D_mm"].dropna().astype(int).sort_values().unique()
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    # --- Plot one line per (D, H) combination ---
    for (D, H), sub in freq_df.groupby(["D_mm", "H_mm"]):
        sub = sub.sort_values("U")
        mk = marker_for_diameter_local(int(D))   # same helper as before

        ax.plot(
            sub["U"].values,
            sub['fft_f0_Hz'].values,
            linestyle="",
            marker=mk,
            color=color_for_height(H),
            alpha=0.9,
        )

    # --- Su Tong reference curve ---
    SuTong_freq = [0.5, 1, 2, 4.25, 7.75, 9]
    SuTong_vel  = [10, 20, 35, 60, 85, 110]
    ax.plot(
        SuTong_vel,
        SuTong_freq,
        color="black",
        linestyle="-",
        marker="o",
        label="Su Tong"
    )

    # --- Legends: reuse the same style you already have ---

    # Height legend is already prepared by make_height_colors
    leg1 = ax.legend(
        handles=height_handles,
        title="Height",
        loc="upper left",
        frameon=True,
    )
    ax.add_artist(leg1)

    # Diameter markers legend: same pattern you showed
    marker_handles = []
    for dval in diameters_present:
        mk = marker_for_diameter_local(dval)
        marker_handles.append(
            Line2D(
                [0], [0],
                marker=mk,
                linestyle="",
                color="0.2",
                markersize=8,
                label=f"D = {dval} mm",
            )
        )

    # Add Su Tong symbol to the second legend
    marker_handles.append(
        Line2D(
            [0], [0],
            marker="o",
            linestyle="-",
            color="black",
            label="Su Tong",
        )
    )

    leg2 = ax.legend(
        handles=marker_handles,
        title="Diameter & Ref.",
        loc="lower right",
        frameon=True,
    )

    # --- Axes / styling ---
    ax.set_xlabel("U [m/s]")
    ax.set_ylabel("FFT peak frequency [Hz]")
    ax.set_title("FFT peak frequency vs Speed U (separated by geometry)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# plot_LBO()

#Bruk hvis du må merke stjernene på nytt
# plot_freq_f0_and_a0('freq_results_log456_last.csv')

# plot_freq_scatter('freq_results_log456_last.csv')
plot_freq_vs_U_by_geom('freq_results_log456_last.csv')

# plot_freq_mean_vs_f0(src_csv='freq_results_log456_last.csv')
# plot_freq_points_vs_f0('freq_results_log456_last.csv')