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

# def plot_fft():
#     base_path = Path(r'data\03_09_D_88mm_350mm')
#     tdms_path = base_path / 'ER1_0,65_Log5_03.09.2025_09.00.39.tdms'
#     mat_path  = base_path / 'Up_8_ERp_0.65_PH2p_0_8_59_1.mat'

#     print("PAIR:", mat_path.name, "<->", tdms_path.name)

#     flow_data_df         = load_tdms_data(tdms_path)
#     pmt_pressure_data_df = load_mat_data(mat_path)

#     # --- sampling interval from timestamps (seconds) ---
#     ts = pmt_pressure_data_df['timestamps']
#     dt = ts.diff().dt.total_seconds().median()
#     if not np.isfinite(dt) or dt <= 0:
#         raise ValueError("Cannot determine a positive sampling interval from timestamps.")
#     fs = 1.0 / dt

#     # --- get signal, sanitize NaNs, detrend (removes DC + slow drift) ---
#     x = pmt_pressure_data_df['PMT'].to_numpy(dtype=float)
#     # Fill NaNs with median to avoid reintroducing DC later
#     med = np.nanmedian(x)
#     x = np.nan_to_num(x, nan=med)

#     # Remove linear trend (use 'constant' if you only want mean removal)
#     x = signal.detrend(x, type='linear')

#     N = x.size

#     # --- windowing (use FFT-style window) ---
#     w = signal.windows.hann(N, sym=False)
#     xw = x * w

#     # --- FFT (one-sided) ---
#     X = np.fft.rfft(xw)
#     f = np.fft.rfftfreq(N, d=dt)

#     # --- amplitude scaling (so a sine of amplitude A shows ~A/2 at its single bin) ---
#     # Coherent gain of the window
#     cg = w.mean()                   # = sum(w)/N
#     amp = (2.0 / N) * np.abs(X) / cg
#     if N % 2 == 0:
#         amp[-1] /= 2.0              # halve Nyquist bin for even N

#     # fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(11, 4))
#     # pmt_pressure_data_df.plot(ax=ax1, x='timestamps', y='PMT', linewidth=1)
#     # ax1.set_title("PMT vs Time")
#     # ax1.set_xlabel("Time")
#     # ax1.set_ylabel("PMT")
#     # ax1.grid(True, which="both", linestyle="--", alpha=0.4)
#     # ax1.legend()

#     # flow_data_df.plot(ax=ax2, x="Time", y=["air_volum_flow", "CH4_volum_flow"], linewidth=1)
#     # ax2.set_title("Mass Flow vs Time")
#     # ax2.set_xlabel("Time")
#     # ax2.set_ylabel("Mass flow")
#     # ax2.grid(True, which="both", linestyle="--", alpha=0.4)
#     # ax2.legend()

#     # fig.tight_layout()
#     # ax2.set_xlim(ax1.get_xlim())
#     # plt.show()


#     # --- plot FFT alone ---
#     plt.figure(figsize=(10, 4))
#     plt.plot(f, amp, label='|X(f)|')
#     plt.title("PMT Amplitude Spectrum")
#     plt.xlabel("Frequency [Hz]")
#     plt.ylabel("Amplitude [a.u.]")
#     plt.grid(True, which="both", linestyle="--", alpha=0.4)
#     plt.xlim(0, fs/2)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# base_path = Path(r'data\03_09_D_88mm_350mm')
# mat_path  = base_path / 'Up_8_ERp_0.65_PH2p_0_8_59_1.mat'
# pmt_pressure_peak_df = load_mat_data(mat_path)

# # ---- Example usage with your plot ----
# def plot_with_peaks(pmt_pressure_data_df):
#     peaks = detect_pmt_peaks(pmt_pressure_data_df)

#     fig, ax = plt.subplots(1, 1, figsize=(11, 3.5))
#     ax.plot(pmt_pressure_data_df['timestamps'], pmt_pressure_data_df['PMT'], label='PMT', linewidth=1)
#     ax.scatter(peaks['timestamp'], peaks['height'], marker='o', color='red', s=18, zorder=3, label='Detected peaks')
#     ax.set_title("PMT vs Time (with peaks)")
#     ax.set_xlabel("Time")
#     ax.set_ylabel("PMT")
#     ax.grid(True, which="both", linestyle="--", alpha=0.4)
#     ax.legend()
#     plt.tight_layout()
#     plt.show()
#     return peaks


# def plot_peak_frequency(peaks_df, ts_col='timestamp', roll_window=5):
#     """
#     Plots instantaneous frequency between detected peaks.
#     Returns a dict of summary stats.
#     """
#     pdf = peaks_df.sort_values(ts_col).reset_index(drop=True).copy()
#     ts = pd.to_datetime(pdf[ts_col])

#     # periods (s) and instantaneous frequency (Hz)
#     dt_s = ts.diff().dt.total_seconds()
#     f = 1.0 / dt_s
#     f = f.replace([np.inf, -np.inf], np.nan)

#     # drop the first NaN (no previous peak)
#     valid = f.notna()
#     t_valid = ts[valid]
#     f_valid = f[valid]

#     if len(f_valid) == 0:
#         raise ValueError("Need at least two peaks to compute frequency.")

#     # rolling median smoother (robust to outliers)
#     f_roll = f_valid.rolling(roll_window, center=True, min_periods=1).median()

#     # summary stats
#     f_mean = float(f_valid.mean())
#     f_std  = float(f_valid.std(ddof=1)) if len(f_valid) > 1 else 0.0
#     f_med  = float(np.median(f_valid))
#     f_mad  = float(1.4826 * np.median(np.abs(f_valid - f_med)))

#     # ---- plot ----
#     fig, ax = plt.subplots(1, 1, figsize=(11, 3.5))
#     ax.plot(t_valid, f_valid, marker='o', ms=3, lw=1, label='Instantaneous freq')
#     ax.plot(t_valid, f_roll, lw=2, label=f'Rolling median (w={roll_window})')
#     ax.axhline(f_mean, lw=1.5, linestyle='--', label=f'Mean = {f_mean:.3f} Hz')
#     ax.fill_between(t_valid, f_mean - f_std, f_mean + f_std, alpha=0.15, label='±1σ')

#     ax.set_title("Instantaneous Peak Frequency")
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Frequency [Hz]")
#     ax.grid(True, which="both", linestyle="--", alpha=0.4)
#     ax.legend()
#     plt.tight_layout()
#     plt.show()

#     return {
#         "n_peaks": int(len(pdf)),
#         "n_intervals": int(len(f_valid)),
#         "freq_mean_Hz": f_mean,
#         "freq_std_Hz": f_std,
#         "freq_median_Hz": f_med,
#         "freq_MAD_Hz": f_mad,
#     }


# # main()

# # plot_fft()

# # peaks = plot_with_peaks(pmt_pressure_peak_df)



# # stats = plot_peak_frequency(peaks)
# # print(stats)

# # stats = peak_period_frequency(peaks)

# # print("Mean frequency (Hz):", stats["freq_mean_Hz"])
# # print("Std of inst. freq (Hz):", stats["freq_std_Hz"])
# #-------------------------------------------------------------------------------------------
# # base_path = Path(r'data\03_09_D_88mm_350mm')
# # tdms_path = base_path / 'ER1_0,65_Log5_03.09.2025_09.00.39.tdms'
# # mat_path = base_path / 'Up_8_ERp_0.65_PH2p_0_8_59_1.mat'

# # base_path = Path(r'data\01_09_D_120mm_260mm')
# # tdms_path = base_path / 'ER1_0,6_log1_01.09.2025_08.42.20.tdms'
# # mat_path = base_path / 'LBO_Sweep_1_8_42_30.mat'

# # # base_path = Path(r'data\03_09_D_88mm_350mm')
# # # tdms_path = base_path / 'ER1_0,65_Log2_03.09.2025_08.46.16.tdms'
# # # mat_path = base_path / 'LBO_Sweep_2_8_46_19.mat'

# # print('base_path: ',base_path)


# # print('-------------------')
# # print('Loading data')
# # print('-------------------')

# # flow_data_df = load_tdms_data(tdms_path)
# # pmt_pressure_data_df = load_mat_data(mat_path)

# # print('-------------------')
# # print('Plotting data')
# # print('-------------------')


# # fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(11, 4))
# # pmt_pressure_data_df.plot(ax=ax1, x='timestamps', y='PMT', linewidth=1)
# # ax1.set_title("PMT vs Time")
# # ax1.set_xlabel("Time")
# # ax1.set_ylabel("PMT")
# # ax1.grid(True, which="both", linestyle="--", alpha=0.4)
# # ax1.legend()

# # pmt_pressure_data_df.plot(ax=ax1, x='timestamps', y='P1', linewidth=1)
# # ax2.set_title("Mass Flow vs Time")
# # ax2.set_xlabel("Time")
# # ax2.set_ylabel("Mass flow")
# # ax2.grid(True, which="both", linestyle="--", alpha=0.4)
# # ax2.legend()

# # fig.tight_layout()
# # ax2.set_xlim(ax1.get_xlim())
# # plt.show()

# # calculate_U_ER(pmt_pressure_data_df,flow_data_df,True)

# # --------------------------------------------------------------------------------------

# # base_path = Path(r'data\03_09_D_88mm_350mm')
# # mat_path  = base_path / 'Up_8_ERp_0.65_PH2p_0_8_59_1.mat'
# # pmt_pressure_peak_df = load_mat_data(mat_path)


# def plot_peak_frequency(peaks_df, ts_col='timestamp', roll_window=5):
#     """
#     Plots instantaneous frequency between detected peaks.
#     Returns a dict of summary stats.
#     """
#     pdf = peaks_df.sort_values(ts_col).reset_index(drop=True).copy()
#     ts = pd.to_datetime(pdf[ts_col])

#     # periods (s) and instantaneous frequency (Hz)
#     dt_s = ts.diff().dt.total_seconds()
#     f = 1.0 / dt_s
#     f = f.replace([np.inf, -np.inf], np.nan)

#     # drop the first NaN (no previous peak)
#     valid = f.notna()
#     t_valid = ts[valid]
#     f_valid = f[valid]

#     if len(f_valid) == 0:
#         raise ValueError("Need at least two peaks to compute frequency.")

#     # rolling median smoother (robust to outliers)
#     f_roll = f_valid.rolling(roll_window, center=True, min_periods=1).median()

#     # summary stats
#     f_mean = float(f_valid.mean())
#     f_std  = float(f_valid.std(ddof=1)) if len(f_valid) > 1 else 0.0
#     f_med  = float(np.median(f_valid))
#     f_mad  = float(1.4826 * np.median(np.abs(f_valid - f_med)))

#     # ---- plot ----
#     fig, ax = plt.subplots(1, 1, figsize=(11, 3.5))
#     ax.plot(t_valid, f_valid, marker='o', ms=3, lw=1, label='Instantaneous freq')
#     ax.plot(t_valid, f_roll, lw=2, label=f'Rolling median (w={roll_window})')
#     ax.axhline(f_mean, lw=1.5, linestyle='--', label=f'Mean = {f_mean:.3f} Hz')
#     ax.fill_between(t_valid, f_mean - f_std, f_mean + f_std, alpha=0.15, label='±1σ')

#     ax.set_title("Instantaneous Peak Frequency")
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Frequency [Hz]")
#     ax.grid(True, which="both", linestyle="--", alpha=0.4)
#     ax.legend()
#     plt.tight_layout()
#     plt.show()

#     return {
#         "n_peaks": int(len(pdf)),
#         "n_intervals": int(len(f_valid)),
#         "freq_mean_Hz": f_mean,
#         "freq_std_Hz": f_std,
#         "freq_median_Hz": f_med,
#         "freq_MAD_Hz": f_mad,
#     }


# # main()

# # plot_fft()

# # peaks = plot_with_peaks(pmt_pressure_peak_df)



# # stats = plot_peak_frequency(peaks)
# # print(stats)

# # stats = peak_period_frequency(peaks)

# # print("Mean frequency (Hz):", stats["freq_mean_Hz"])
# # print("Std of inst. freq (Hz):", stats["freq_std_Hz"])

# # ---------------------------------------------------------------------------------------------------
# # --- Test Signal ---
# fs = 1000        # Sampling frequency [Hz]
# T = 1.0          # Duration [s]
# f_sig = 50       # Signal frequency [Hz]
# amplitude = 1.0
# noise_level = 0.5  # set >0 for noise, e.g. 0.2

# # --- Time vector ---
# t_sin = np.arange(0, T, 1/fs)

# # --- Signal: sine wave + optional noise ---
# y_sin = amplitude*2 * np.sin(2 * np.pi * f_sig * t_sin) + noise_level * np.random.randn(len(t_sin)) + amplitude * np.sin(2 * np.pi * 100 * t_sin)
# # ---------------------------------------------------------------------------------------------------
# # calculate_fft(y_sin,t_sin,fs)
# # ---------------------------------------------------------------------------------------------------

# def plot_fft_draft():
#     base_path = Path(r'data\03_09_D_88mm_350mm')
#     tdms_path = base_path / 'ER1_0,65_Log5_03.09.2025_09.00.39.tdms'
#     mat_path  = base_path / 'Up_8_ERp_0.65_PH2p_0_8_59_1.mat'

#     print("PAIR:", mat_path.name, "<->", tdms_path.name)

#     flow_data_df         = load_tdms_data(tdms_path)
#     pmt_pressure_data_df = load_mat_data(mat_path)

#     # --- sampling interval from timestamps (seconds) ---
#     ts = pmt_pressure_data_df['timestamps']
#     dt = ts.diff().dt.total_seconds().median()
#     if not np.isfinite(dt) or dt <= 0:
#         raise ValueError("Cannot determine a positive sampling interval from timestamps.")
#     fs = 1.0 / dt

#     # --- get signal, sanitize NaNs, detrend (removes DC + slow drift) ---
#     x = pmt_pressure_data_df['PMT'].to_numpy(dtype=float)
#     # Fill NaNs with median to avoid reintroducing DC later
#     med = np.nanmedian(x)
#     x = np.nan_to_num(x, nan=med)

#     # Remove linear trend (use 'constant' if you only want mean removal)
#     x = signal.detrend(x, type='linear')

#     N = x.size

#     # --- windowing (use FFT-style window) ---
#     w = signal.windows.hann(N, sym=False)
#     xw = x * w

#     # --- FFT (one-sided) ---
#     X = np.fft.rfft(xw)
#     f = np.fft.rfftfreq(N, d=dt)

#     # --- amplitude scaling (so a sine of amplitude A shows ~A/2 at its single bin) ---
#     # Coherent gain of the window
#     cg = w.mean()                   # = sum(w)/N
#     amp = (2.0 / N) * np.abs(X) / cg
#     if N % 2 == 0:
#         amp[-1] /= 2.0              # halve Nyquist bin for even N

#     # fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(11, 4))
#     # pmt_pressure_data_df.plot(ax=ax1, x='timestamps', y='PMT', linewidth=1)
#     # ax1.set_title("PMT vs Time")
#     # ax1.set_xlabel("Time")
#     # ax1.set_ylabel("PMT")
#     # ax1.grid(True, which="both", linestyle="--", alpha=0.4)
#     # ax1.legend()

#     # flow_data_df.plot(ax=ax2, x="Time", y=["air_volum_flow", "CH4_volum_flow"], linewidth=1)
#     # ax2.set_title("Mass Flow vs Time")
#     # ax2.set_xlabel("Time")
#     # ax2.set_ylabel("Mass flow")
#     # ax2.grid(True, which="both", linestyle="--", alpha=0.4)
#     # ax2.legend()

#     # fig.tight_layout()
#     # ax2.set_xlim(ax1.get_xlim())
#     # plt.show()


#     # --- plot FFT alone ---
#     plt.figure(figsize=(10, 4))
#     plt.plot(f, amp, label='|X(f)|')
#     plt.title("PMT Amplitude Spectrum")
#     plt.xlabel("Frequency [Hz]")
#     plt.ylabel("Amplitude [a.u.]")
#     plt.grid(True, which="both", linestyle="--", alpha=0.4)
#     plt.xlim(0, fs/2)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# def calculate_fft_draft2(input_signal, input_time,fft_fs: float):

#     N = len(input_signal)

#     w = signal.windows.hann(N, sym=False)
#     input_signal_w = input_signal*w

#     Nfft  = sfft.next_fast_len(N)

#     fft_output = sfft.rfft(input_signal, n=Nfft)
#     freq  = sfft.rfftfreq(N, d=1/fft_fs)

#     amp = (2.0 / len(input_signal)) * np.abs(fft_output)  # double everything and normalize by N
#     if len(input_signal) % 2 == 0:                # even N → Nyquist bin exists at the end
#         amp[-1] /= 2                   # undo doubling for Nyquist
#     amp[0] /= 2                        # undo doubling for DC

#     window_length = len(input_signal) // 16
#     window_overlap = window_length // 2

#     f_welch, Pxx = signal.welch(input_signal,fs=fft_fs,window='hann',nperseg=window_length,noverlap=window_overlap)

#     # --- Plots ---
#     fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(10,5))

#     ax1.plot(input_time,input_signal)
#     ax1.set_title('PMT data')
#     ax1.set_ylabel('PMT signal')
#     ax1.set_xlabel('Time')

#     ax2.plot(freq,amp)
#     ax2.set_xlim(1,500)
#     ax2.set_title('FFT data')
#     ax2.set_ylabel('FFT of PMT signal')
#     ax2.set_xlabel('Frequency [Hz]')

#     ax3.plot(f_welch,Pxx)
#     ax3.set_title('Power spectral density')
#     ax3.set_ylabel('Power spectral density')
#     ax3.set_xlabel('Frequency [Hz]')

#     plt.grid(True, which="both", linestyle="--", alpha=0.4)
#     plt.tight_layout()


# # base_path = Path(r'data\03_09_D_88mm_350mm')
# # tdms_path = base_path / 'ER1_0,65_Log5_03.09.2025_09.00.39.tdms'
# # mat_path  = base_path / 'Up_8_ERp_0.65_PH2p_0_8_59_1.mat'


# base_path = Path(r'data\03_09_D_88mm_350mm')
# tdms_path = base_path / 'ER1_0,71_Log4_04.09.2025_09.10.18.tdms'
# mat_path  = base_path / 'Up_20_ERp_0.71_PH2p_0_9_10_22.mat'


# print("PAIR:", mat_path.name, "<->", tdms_path.name)

# # flow_data_df         = load_tdms_data(tdms_path)
# pmt_df = load_mat_data(mat_path)

# ts_pmt = pmt_df['timestamps']
# d = ts_pmt.diff().dt.total_seconds().median()
# fs_pmt = 1.0 / d

# pmt_raw = pmt_df['PMT']
# pmt = pmt_raw - np.mean(pmt_raw)

# fft_stats = calculate_fft(pmt,ts_pmt,mat_path.stem, tdms_path.stem, mat_path.parent.name)
# f0 = fft_stats["f0_Hz"]
# if np.isnan(f0):
#     f0 = None
# print(f"FFT dominant ≈ {f0 if f0 is not None else float('nan'):.3f} Hz "
#     f"(fs={fft_stats['fs_Hz']:.3f} Hz, N={fft_stats['N']})")

x = 1000000
N    = round(x/16)
w    = signal.windows.hann(N, sym=False)
w2 = np.array([])
for i in range(16):
    print('idx: ',i)
    w2 = np.append(w2,w)
    print(w2)
    print(len(w2))

