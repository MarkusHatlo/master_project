from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import re


from scipy.signal import find_peaks, savgol_filter
from scipy import signal
from datetime import datetime, timedelta
from pathlib import Path
from nptdms import TdmsFile
from collections import defaultdict

def plot_fft():
    base_path = Path(r'data\03_09_D_88mm_350mm')
    tdms_path = base_path / 'ER1_0,65_Log5_03.09.2025_09.00.39.tdms'
    mat_path  = base_path / 'Up_8_ERp_0.65_PH2p_0_8_59_1.mat'

    print("PAIR:", mat_path.name, "<->", tdms_path.name)

    flow_data_df         = load_tdms_data(tdms_path)
    pmt_pressure_data_df = load_mat_data(mat_path)

    # --- sampling interval from timestamps (seconds) ---
    ts = pmt_pressure_data_df['timestamps']
    dt = ts.diff().dt.total_seconds().median()
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Cannot determine a positive sampling interval from timestamps.")
    fs = 1.0 / dt

    # --- get signal, sanitize NaNs, detrend (removes DC + slow drift) ---
    x = pmt_pressure_data_df['PMT'].to_numpy(dtype=float)
    # Fill NaNs with median to avoid reintroducing DC later
    med = np.nanmedian(x)
    x = np.nan_to_num(x, nan=med)

    # Remove linear trend (use 'constant' if you only want mean removal)
    x = signal.detrend(x, type='linear')

    N = x.size

    # --- windowing (use FFT-style window) ---
    w = signal.windows.hann(N, sym=False)
    xw = x * w

    # --- FFT (one-sided) ---
    X = np.fft.rfft(xw)
    f = np.fft.rfftfreq(N, d=dt)

    # --- amplitude scaling (so a sine of amplitude A shows ~A/2 at its single bin) ---
    # Coherent gain of the window
    cg = w.mean()                   # = sum(w)/N
    amp = (2.0 / N) * np.abs(X) / cg
    if N % 2 == 0:
        amp[-1] /= 2.0              # halve Nyquist bin for even N

    # fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(11, 4))
    # pmt_pressure_data_df.plot(ax=ax1, x='timestamps', y='PMT', linewidth=1)
    # ax1.set_title("PMT vs Time")
    # ax1.set_xlabel("Time")
    # ax1.set_ylabel("PMT")
    # ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    # ax1.legend()

    # flow_data_df.plot(ax=ax2, x="Time", y=["air_volum_flow", "CH4_volum_flow"], linewidth=1)
    # ax2.set_title("Mass Flow vs Time")
    # ax2.set_xlabel("Time")
    # ax2.set_ylabel("Mass flow")
    # ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    # ax2.legend()

    # fig.tight_layout()
    # ax2.set_xlim(ax1.get_xlim())
    # plt.show()


    # --- plot FFT alone ---
    plt.figure(figsize=(10, 4))
    plt.plot(f, amp, label='|X(f)|')
    plt.title("PMT Amplitude Spectrum")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [a.u.]")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.xlim(0, fs/2)
    plt.legend()
    plt.tight_layout()
    plt.show()

base_path = Path(r'data\03_09_D_88mm_350mm')
mat_path  = base_path / 'Up_8_ERp_0.65_PH2p_0_8_59_1.mat'
pmt_pressure_peak_df = load_mat_data(mat_path)

# ---- Example usage with your plot ----
def plot_with_peaks(pmt_pressure_data_df):
    peaks = detect_pmt_peaks(pmt_pressure_data_df)

    fig, ax = plt.subplots(1, 1, figsize=(11, 3.5))
    ax.plot(pmt_pressure_data_df['timestamps'], pmt_pressure_data_df['PMT'], label='PMT', linewidth=1)
    ax.scatter(peaks['timestamp'], peaks['height'], marker='o', color='red', s=18, zorder=3, label='Detected peaks')
    ax.set_title("PMT vs Time (with peaks)")
    ax.set_xlabel("Time")
    ax.set_ylabel("PMT")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return peaks


def plot_peak_frequency(peaks_df, ts_col='timestamp', roll_window=5):
    """
    Plots instantaneous frequency between detected peaks.
    Returns a dict of summary stats.
    """
    pdf = peaks_df.sort_values(ts_col).reset_index(drop=True).copy()
    ts = pd.to_datetime(pdf[ts_col])

    # periods (s) and instantaneous frequency (Hz)
    dt_s = ts.diff().dt.total_seconds()
    f = 1.0 / dt_s
    f = f.replace([np.inf, -np.inf], np.nan)

    # drop the first NaN (no previous peak)
    valid = f.notna()
    t_valid = ts[valid]
    f_valid = f[valid]

    if len(f_valid) == 0:
        raise ValueError("Need at least two peaks to compute frequency.")

    # rolling median smoother (robust to outliers)
    f_roll = f_valid.rolling(roll_window, center=True, min_periods=1).median()

    # summary stats
    f_mean = float(f_valid.mean())
    f_std  = float(f_valid.std(ddof=1)) if len(f_valid) > 1 else 0.0
    f_med  = float(np.median(f_valid))
    f_mad  = float(1.4826 * np.median(np.abs(f_valid - f_med)))

    # ---- plot ----
    fig, ax = plt.subplots(1, 1, figsize=(11, 3.5))
    ax.plot(t_valid, f_valid, marker='o', ms=3, lw=1, label='Instantaneous freq')
    ax.plot(t_valid, f_roll, lw=2, label=f'Rolling median (w={roll_window})')
    ax.axhline(f_mean, lw=1.5, linestyle='--', label=f'Mean = {f_mean:.3f} Hz')
    ax.fill_between(t_valid, f_mean - f_std, f_mean + f_std, alpha=0.15, label='±1σ')

    ax.set_title("Instantaneous Peak Frequency")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency [Hz]")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return {
        "n_peaks": int(len(pdf)),
        "n_intervals": int(len(f_valid)),
        "freq_mean_Hz": f_mean,
        "freq_std_Hz": f_std,
        "freq_median_Hz": f_med,
        "freq_MAD_Hz": f_mad,
    }


# main()

# plot_fft()

# peaks = plot_with_peaks(pmt_pressure_peak_df)



# stats = plot_peak_frequency(peaks)
# print(stats)

# stats = peak_period_frequency(peaks)

# print("Mean frequency (Hz):", stats["freq_mean_Hz"])
# print("Std of inst. freq (Hz):", stats["freq_std_Hz"])
#-------------------------------------------------------------------------------------------
# base_path = Path(r'data\03_09_D_88mm_350mm')
# tdms_path = base_path / 'ER1_0,65_Log5_03.09.2025_09.00.39.tdms'
# mat_path = base_path / 'Up_8_ERp_0.65_PH2p_0_8_59_1.mat'

# base_path = Path(r'data\01_09_D_120mm_260mm')
# tdms_path = base_path / 'ER1_0,6_log1_01.09.2025_08.42.20.tdms'
# mat_path = base_path / 'LBO_Sweep_1_8_42_30.mat'

# # base_path = Path(r'data\03_09_D_88mm_350mm')
# # tdms_path = base_path / 'ER1_0,65_Log2_03.09.2025_08.46.16.tdms'
# # mat_path = base_path / 'LBO_Sweep_2_8_46_19.mat'

# print('base_path: ',base_path)


# print('-------------------')
# print('Loading data')
# print('-------------------')

# flow_data_df = load_tdms_data(tdms_path)
# pmt_pressure_data_df = load_mat_data(mat_path)

# print('-------------------')
# print('Plotting data')
# print('-------------------')


# fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(11, 4))
# pmt_pressure_data_df.plot(ax=ax1, x='timestamps', y='PMT', linewidth=1)
# ax1.set_title("PMT vs Time")
# ax1.set_xlabel("Time")
# ax1.set_ylabel("PMT")
# ax1.grid(True, which="both", linestyle="--", alpha=0.4)
# ax1.legend()

# pmt_pressure_data_df.plot(ax=ax1, x='timestamps', y='P1', linewidth=1)
# ax2.set_title("Mass Flow vs Time")
# ax2.set_xlabel("Time")
# ax2.set_ylabel("Mass flow")
# ax2.grid(True, which="both", linestyle="--", alpha=0.4)
# ax2.legend()

# fig.tight_layout()
# ax2.set_xlim(ax1.get_xlim())
# plt.show()

# calculate_U_ER(pmt_pressure_data_df,flow_data_df,True)