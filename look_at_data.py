from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import re
import time

from scipy import fft as sfft
from scipy.signal import find_peaks, savgol_filter
from scipy import signal
from datetime import datetime, timedelta
from pathlib import Path
from nptdms import TdmsFile
from collections import defaultdict

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


def look_at_pmt_data(x, ts, col='PMT',
                     smooth_ms=100,         # small smoothing for noise
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

    ax1.plot(x,color='black')
    ax1.plot(baseline,color='red')

    ax2.plot(y,color='blue')
    ax2.axhline(0)

    ax3.plot(y_smooth)
    plt.show()

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
    resolution = fft_fs/total_N
    print('The total number of points, N:', total_N)
    print('Sampling frequency:', fft_fs)
    print('The resolution is:', resolution)

    fig, ax1 = plt.subplots(1,1)
    ax1.plot(input_time, input_signal, label='Raw (DC removed)')
    ax1.set_title('PMT data')
    ax1.set_ylabel('PMT signal')
    ax1.set_xlabel('Time')
    ax1.grid(True, linestyle="--", alpha=0.3)
    plt.show()


base_path = Path(r'data\03_09_D_88mm_350mm')
mat  = base_path / 'LBO_Sweep_1_8_39_29.mat'
tdms = base_path / "ER1_0,65_Log1_03.09.2025_08.39.27.tdms"

print('Making the dataframes')
pmt_pressure_dataFrame = load_mat_data(mat)
flow_dataFrame = load_tdms_data(tdms)


print('Defining calculation windows')
window_start, window_stop = calculating_window(pmt_pressure_dataFrame,flow_dataFrame)
pmt_window   = pmt_pressure_dataFrame['PMT'].iloc[window_start:window_stop].to_numpy(float)
time_window  = pmt_pressure_dataFrame['timestamps'].iloc[window_start:window_stop]

# pmt_window   = pmt_pressure_dataFrame['PMT']
# time_window  = pmt_pressure_dataFrame['timestamps']

print('Detecting peaks')
peaks = look_at_pmt_data(pmt_window, time_window)
# print('Checking freq resolution')
# check_freq_resolution(pmt_window,time_window)