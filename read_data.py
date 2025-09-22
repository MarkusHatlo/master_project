import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from datetime import datetime, timedelta

from pathlib import Path
from nptdms import TdmsFile

def find_channel_names():
    print([g.name for g in tdms.groups()])
    for g in tdms.groups():
        print(g.name, [c.name for c in g.channels()])

def plot_massflows():
    # ----- plot (single subplot/axes) -----
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))  # using the subplot API as requested
    flow_df.plot(ax=ax, x="Time", y=["air mass flow", "CH4 mass flow"], linewidth=1)

    ax.set_title("Mass Flow vs Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mass flow")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    plt.show()

def plot_pmt(show_plot):
    # Choose a threshold for "near zero" (1% of max is a decent default)
    thr = 0
    # Find first index where we cross from >thr to <=thr
    cross_idx_pmt = np.where(pmt_pressure_df['PMT'] <= thr)[0]
    i_cross_pmt = int(cross_idx_pmt[0])
    print("First near-zero crossing index:", i_cross_pmt)

    nearest_idx_flow = (flow_df['Time'] - pmt_pressure_df['timestamps'][i_cross_pmt]).abs().idxmin()
    cross_flow_time_value = flow_df.loc[nearest_idx_flow, 'Time']
    print("Nearest near-zero crossing index for flow:", nearest_idx_flow, cross_flow_time_value)

    start_idx_flow = ((flow_df['Time'] - pmt_pressure_df['timestamps'][0]).abs().idxmin())
    start_flow_time_value = flow_df.loc[start_idx_flow, 'Time']
    print("First index flow:", start_idx_flow, start_flow_time_value)

    if show_plot:

        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(11, 4))
        pmt_pressure_df.plot(ax=ax1, x='timestamps', y='PMT', linewidth=1)
        ax1.axvline(pmt_pressure_df['timestamps'][i_cross_pmt], color='red')
        ax1.set_title("PMT vs Time")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("PMT")
        ax1.grid(True, which="both", linestyle="--", alpha=0.4)
        ax1.legend()

        flow_df.plot(ax=ax2, x="Time", y=["air mass flow", "CH4 mass flow"], linewidth=1)
        ax2.axvline(cross_flow_time_value, color='red')
        ax2.set_title("Mass Flow vs Time")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Mass flow")
        ax2.grid(True, which="both", linestyle="--", alpha=0.4)
        ax2.legend()

        fig.tight_layout()
        ax2.set_xlim(ax1.get_xlim())
        plt.show()

base_path = Path(r'D:\202508Experiment_data_logging\03_09_D_88mm_350mm')
tdms_path = base_path / 'ER1_0,65_Log2_03.09.2025_08.46.16.tdms'

assert tdms_path.exists(), f"Not found: {tdms_path}"

with TdmsFile.read(tdms_path) as tdms:   # use .open(...) for very large files
    grp = tdms['Data']

    air_volum_flow = grp['Z - Mass flow'][:]
    CH4_volum_flow = grp['X - Mass flow'][:]
    date_time_raw = grp['Time'][:]
    date_time = pd.to_datetime(date_time_raw,utc=True)  # tz-aware UTC
    date_time = date_time.tz_convert("Europe/Oslo")


    flow_df = pd.DataFrame({
            'Time' : date_time,
            'air mass flow' : air_volum_flow,
            'CH4 mass flow' : CH4_volum_flow
    })


mat_time_path = base_path / 'LBO_Sweep_2_8_46_19_posix.mat'

m = sio.loadmat(mat_time_path, squeeze_me=True, struct_as_record=False)
posix = float(m['data'].timestamp_fast_posix)
start_time_pmt = datetime(1970,1,1) + timedelta(seconds=posix)


mat_path = base_path / 'LBO_Sweep_2_8_46_19.mat'

m = sio.loadmat(mat_path, squeeze_me=True, simplify_cells=True)
mat_data = m['data']            # now a plain dict (SciPy ≥1.7)
# print(list(mat_data.keys()))

t_rel = np.asarray(mat_data['time_fast'], dtype=float).ravel()  # seconds
timestamps = pd.to_datetime(start_time_pmt, utc=True) + pd.to_timedelta(t_rel, unit='s') - pd.Timedelta(hours=1)
timestamps = timestamps.tz_convert("Europe/Oslo")
timestamps_pmt = timestamps.to_pydatetime()  # ndarray of datetime objects

pmt_pressure_df = pd.DataFrame({
    # 'timestamp_fast': pd.to_datetime(start_time) + pd.to_timedelta(time_fast, unit="s"),
    'time_fast' : np.asarray(mat_data['time_fast'], dtype=float).ravel(),
    'timestamps' : timestamps_pmt,
    'PMT'  : np.asarray(mat_data['PMT_OH_1'], dtype=float).ravel(),
    'Cam_trig'  : np.asarray(mat_data['Cam_trig'], dtype=float).ravel(),
    'P1'        : np.asarray(mat_data['P1'], dtype=float).ravel(),
    'P2'        : np.asarray(mat_data['P2'], dtype=float).ravel(),
    'P3'        : np.asarray(mat_data['P3'], dtype=float).ravel(),
    'Pref'      : np.asarray(mat_data['Pref'], dtype=float).ravel(),
})

plot_pmt(True)
# plot_massflows()