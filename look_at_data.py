from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import re
import time

from scipy import fft as sfft
from scipy.signal import find_peaks, savgol_filter,butter, filtfilt
from scipy import signal
from datetime import datetime, timedelta
from pathlib import Path
from nptdms import TdmsFile
from collections import defaultdict

def _secs_since_midnight(h: int, m: int, s: int) -> int:
    return h*3600 + m*60 + s

_mat_time_re = re.compile(r'_(\d{1,2})_(\d{1,2})_(\d{1,2})\.mat$', re.IGNORECASE)
def parse_mat_time_seconds(name: str) -> int | None:
    """
    Extract HH_MM_SS from MAT names like ..._11_24_6.mat → 11:24:06.
    Returns seconds since midnight, or None if not matched.
    """
    m = _mat_time_re.search(name)
    if not m:
        return None
    h, mi, s = map(int, m.groups())
    return _secs_since_midnight(h, mi, s)

_tdms_time_re = re.compile(r'_(\d{2}\.\d{2}\.\d{4})_(\d{2})\.(\d{2})\.(\d{2})\.tdms$',  # ..._DD.MM.YYYY_HH.MM.SS.tdms
    re.IGNORECASE)
def parse_tdms_time_seconds(name: str) -> int | None:
    """
    Extract time from TDMS names like ..._03.09.2025_11.24.02.tdms → 11:24:02.
    Returns seconds since midnight, or None if not matched.
    """
    m = _tdms_time_re.search(name)
    if not m:
        return None
    _, hh, mm, ss = m.groups()
    return _secs_since_midnight(int(hh), int(mm), int(ss))

def iter_data_files(base_path: Path, include_subfolders=False):
    """
    Yields every .tdms and .mat file under base_path.
    Set include_subfolders=True to recurse into subfolders.
    """
    base = Path(base_path)
    patterns = ("*.tdms", "*.mat")

    # Choose globber based on recursion
    globber = base.rglob if include_subfolders else base.glob

    # Use a set to avoid duplicates, then sort for stable order
    files = {
        p.resolve()
        for pat in patterns
        for p in globber(pat)
        if p.is_file()
    }
    # Sort by extension then name (case-insensitive)
    return sorted(files, key=lambda p: (p.suffix.lower(), p.name.lower()))

def pair_mat_tdms(files: List[Path], *, tolerance_seconds=20, group_by_dir=True):
    """
    Pair .mat with the closest-in-time .tdms, within tolerance.
    If group_by_dir=True, only pair files that share the same parent directory.
    Returns (pairs, unmatched_mats, unmatched_tdms).
    """
    # Group files by directory if requested, else put all in one bucket
    buckets = defaultdict(list)
    for p in files:
        key = p.parent if group_by_dir else "_all_"
        buckets[key].append(p)

    all_pairs = []
    all_unmatched_mats = []
    all_unmatched_tdms = []

    for key, bucket in buckets.items():
        mats = []
        tdms = []

        for p in bucket:
            if p.suffix.lower() == ".mat":
                t = parse_mat_time_seconds(p.name)
                if t is not None:
                    mats.append((p, t))
            elif p.suffix.lower() == ".tdms":
                t = parse_tdms_time_seconds(p.name)
                if t is not None:
                    tdms.append((p, t))

        # Sort by time to make greedy matching stable
        mats.sort(key=lambda x: x[1])
        tdms.sort(key=lambda x: x[1])

        used_mat = set()
        used_tdms = set()

        # Greedy: for each tdms, pick the nearest unused mat within tolerance
        for j, (tdms_path, t_tdms) in enumerate(tdms):
            best_i = None
            best_diff = None
            for i, (mat_path, t_mat) in enumerate(mats):
                if i in used_mat:
                    continue
                diff = abs(t_mat - t_tdms)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_i = i
            if best_i is not None and best_diff is not None and best_diff <= tolerance_seconds:
                used_tdms.add(j)
                used_mat.add(best_i)
                all_pairs.append((mats[best_i][0], tdms_path))  # (mat, tdms)

        # Collect unmatched
        all_unmatched_mats.extend(m for k, (m, _) in enumerate(mats) if k not in used_mat)
        all_unmatched_tdms.extend(t for k, (t, _) in enumerate(tdms) if k not in used_tdms)

    return all_pairs, all_unmatched_mats, all_unmatched_tdms

def load_mat_data(mat_path: Path):
    #load the mat data

    m = sio.loadmat(mat_path, squeeze_me=True, simplify_cells=True)
    mat_data = m['data']            # now a plain dict (SciPy ≥1.7)
    # print(list(mat_data.keys()))

    posix = float(mat_data['timestamp_fast_posix'])
    start_time_pmt = datetime(1970,1,1) + timedelta(seconds=posix)
    t_rel = np.asarray(mat_data['time_fast'], dtype=float).ravel()  # seconds
    timestamps = pd.to_datetime(start_time_pmt, utc=True) + pd.to_timedelta(t_rel, unit='s')# - pd.Timedelta(hours=1)
    timestamps = timestamps.tz_convert("Europe/Oslo")
    timestamps_pmt = timestamps.to_pydatetime()  # ndarray of datetime objects

    pmt_pressure_df = pd.DataFrame({
        'time_fast' : np.asarray(mat_data['time_fast'], dtype=float).ravel(),
        'timestamps' : timestamps_pmt,
        'PMT'  : np.asarray(mat_data['PMT_OH_1'], dtype=float).ravel(),
        'Cam_trig'  : np.asarray(mat_data['Cam_trig'], dtype=float).ravel(),
        'P1'        : np.asarray(mat_data['P1'], dtype=float).ravel(),
        'P2'        : np.asarray(mat_data['P2'], dtype=float).ravel(),
        'P3'        : np.asarray(mat_data['P3'], dtype=float).ravel(),
        'Pref'      : np.asarray(mat_data['Pref'], dtype=float).ravel(),
    })


    return pmt_pressure_df

def load_tdms_data(tdms_path: Path):
    #load the tdms data
    assert tdms_path.exists(), f"Not found: {tdms_path}"

    with TdmsFile.read(tdms_path) as tdms:   # use .open(...) for very large files
        grp = tdms['Data']

        air_volum_flow = grp['Z - Mass flow'][:]
        CH4_volum_flow = grp['X - Mass flow'][:]
        date_time_raw = grp['Time'][:]
        date_time = pd.to_datetime(date_time_raw,utc=True)  # tz-aware UTC
        date_time = date_time.tz_convert("Europe/Oslo")
        date_time = date_time - pd.Timedelta(hours=1)


        flow_df = pd.DataFrame({
                'Time' : date_time,
                'air_volum_flow' : air_volum_flow,
                'CH4_volum_flow' : CH4_volum_flow
        })

        return flow_df

def look_at_pmt_data(x, ts, matFileName: str, tdmsFileName: str, folderName: str,
                     smooth_ms=500,         # small smoothing for noise
                     baseline_ms=1000,      # rolling-median baseline removal
                     min_distance_s=0.30,  # refractory time between peaks
                     min_width_ms=10,      # discard ultra-narrow blips
                     prominence_sigma=7.0, # how strong above noise
                     rel_height=0.5):      # width at 50% prominence

    dt = ts.diff().dt.total_seconds().median()
    fs = 1.0 / dt

    # --- baseline remove with rolling median (robust to outliers) ---
    win_baseline = max(3, int(round(baseline_ms/1000 * fs)))
    baseline = pd.Series(x).rolling(win_baseline, center=True, min_periods=1).median().to_numpy()
    y = x - baseline
    w = signal.windows.hann(len(y), sym=False)
    seg_y = y*w

    
    w = max(3, int(round(smooth_ms/1000 * fs)) | 1)  # odd
    y_smooth = savgol_filter(y, window_length=w, polyorder=2, mode='interp')


    fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(12, 5))
    # fig, (ax1) = plt.subplots(1, 1, figsize=(7,7))


    ax1.plot(x,color='black')
    ax1.plot(baseline,color='red')
    # ax1.set_xlim(int(9.7e5),int(1e6))

    # x_dot = [9.845e5,9.83e5,9.88e5,9.8e5,9.896e5]
    # y_dot = [3,1.5,1.5,0.51,0.6]
    # ax1.plot(x_dot, y_dot, 'o', color='red')

    # ax2.plot(pmt_pressure_dataFrame['Cam_trig'])
    ax2.plot(y,color='blue')
    ax2.axhline(0)

    ax3.plot(y_smooth)

    plt.tight_layout()
    plt.show()

    # # --- save figure ---
    # picture_path = Path('pictures')
    # out_dir = picture_path / 'only_the_pmt' / folderName
    # out_dir.mkdir(parents=True, exist_ok=True)
    # out_path = out_dir / f'{matFileName}_and_{tdmsFileName}_FFT.png'
    # fig.savefig(out_path, dpi=300, bbox_inches='tight')
    # plt.close(fig)

crossing_threshold = 0
def calculating_window(df, flow_df):
        steady_state_air = flow_df['air_volum_flow'].iloc[:10].mean()
        print('steady_state_air', steady_state_air)
        ramping_idx_air = np.where(flow_df['air_volum_flow'] > 1.02*steady_state_air)[0]
        i_ramping_air = int(ramping_idx_air[0])
        ramp_time = flow_df['Time'].iloc[i_ramping_air]
        start_idx_pmt = (df['timestamps'] - ramp_time).abs().idxmin()

        all_cross_idxs = np.where(df['PMT'] <= crossing_threshold)[0]
        cross_after_ramp = all_cross_idxs[all_cross_idxs > start_idx_pmt]
        stop_idx = int(cross_after_ramp[0])

        print('Start idx', start_idx_pmt,'Stop idx', stop_idx)
        return start_idx_pmt, stop_idx

def check_freq_resolution(input_signal, input_time):
    total_N = len(input_signal)
    d = input_time.diff().dt.total_seconds().median()
    fft_fs = 1.0 / d  # Hz
    resolution = fft_fs/(total_N // 6)
    print('The total number of points, N:', total_N)
    print('Sampling frequency:', fft_fs)
    print('The resolution is:', resolution)

    # fig, ax1 = plt.subplots(1,1)
    # ax1.plot(input_time, input_signal, label='Raw (DC removed)')
    # ax1.set_title('PMT data')
    # ax1.set_ylabel('PMT signal')
    # ax1.set_xlabel('Time')
    # ax1.grid(True, linestyle="--", alpha=0.3)
    # plt.show()

# Single data
# ------------------------------------------------------------------------------------------------------
base_path = Path(r'data_handpicked\03_09_D_88mm_350mm')
mat  = base_path / 'LBO_Sweep_1_8_39_29.mat'
tdms = base_path / "ER1_0,65_Log1_03.09.2025_08.39.27.tdms"

# tdms = base_path / 'ER1_0,7_log4_29.08.2025_12.52.19.tdms'
# mat  = base_path / 'Up_15_ERp_0.65_PH2p_0_12_52_22.mat'

print('Making the dataframes')
pmt_pressure_dataFrame = load_mat_data(mat)
# flow_dataFrame = load_tdms_data(tdms)


# print('Defining calculation windows')
# window_start, window_stop = calculating_window(pmt_pressure_dataFrame,flow_dataFrame)
# pmt_window   = pmt_pressure_dataFrame['PMT'].iloc[window_start:window_stop].to_numpy(float)
# time_window  = pmt_pressure_dataFrame['timestamps'].iloc[window_start:window_stop]

pmt_window   = pmt_pressure_dataFrame['PMT']
time_window  = pmt_pressure_dataFrame['timestamps']


# print('Detecting peaks')
# peaks = look_at_pmt_data(pmt_window, time_window)
# print('Checking freq resolution')
# check_freq_resolution(pmt_window,time_window)
peaks = look_at_pmt_data(pmt_window, time_window, mat.stem, tdms.stem, mat.parent.name)
# ------------------------------------------------------------------------------------------------------

# # Loop
# base_path = Path('only_good_data')
# files = iter_data_files(base_path, True)

# #Find the pairs in the code
# pairs, um_mats, um_tdms = pair_mat_tdms(
#     files,
#     tolerance_seconds=55,   # tweak if needed
#     group_by_dir=True
# )
# total_files = len(pairs)

# for file_idx, (mat, tdms) in enumerate(pairs):
#     print(f'Files processed {file_idx+1}/{total_files}')
#     print("PAIR:", mat.name, "<->", tdms.name)

#     flow_dataFrame = load_tdms_data(tdms)
#     pmt_pressure_dataFrame = load_mat_data(mat)

#     pmt_window   = pmt_pressure_dataFrame['PMT']
#     time_window  = pmt_pressure_dataFrame['timestamps']

#     print('Detecting peaks')
#     peaks = look_at_pmt_data(pmt_window, time_window, mat.stem, tdms.stem, mat.parent.name)
