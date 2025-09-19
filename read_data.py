import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

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

def plot_pmt(pmt_signal):
    # Choose a threshold for "near zero" (1% of max is a decent default)
    thr = 0
    # Find first index where we cross from >thr to <=thr
    cross_idx = np.where((pmt_signal[:-1] > thr) & (pmt_signal[1:] <= thr))[0]
    i_cross = int(cross_idx[0] + 1) if len(cross_idx) else None
    print("First near-zero crossing index:", i_cross)

    fig, ax = plt.subplots(1, 1, figsize=(11, 4))  # using the subplot API as requested
    pmt_pressure_df.plot(ax=ax, x='time_fast', y='PMT', linewidth=1)
    ax.axvline(i_cross)

    ax.set_title("PMT vs Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("PMT")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    plt.show()

base_path = Path(r'G:\202508Experiment_data_logging\03_09_D_88mm_350mm')
tdms_path = base_path / 'ER1_0,65_Log2_03.09.2025_08.46.16.tdms'

assert tdms_path.exists(), f"Not found: {tdms_path}"

with TdmsFile.read(tdms_path) as tdms:   # use .open(...) for very large files
    grp = tdms['Data']

    air_volum_flow = grp['Z - Mass flow'][:]
    CH4_volum_flow = grp['X - Mass flow'][:]
    date_time_raw = grp['Time'][:]
    date_time = pd.to_datetime(date_time_raw, format="%Y-%m-%d %H:%M:%S.%f")

    flow_df = pd.DataFrame({
            'Time' : date_time,
            'air mass flow' : air_volum_flow,
            'CH4 mass flow' : CH4_volum_flow
    })

mat_path = base_path / 'LBO_Sweep_2_8_46_19.mat'

m = sio.loadmat(mat_path, squeeze_me=True, simplify_cells=True)
mat_data = m['data']            # now a plain dict (SciPy ≥1.7)
print(list(mat_data.keys()))

start_time_pmt = mat_data['timestamp_fast'] 
# time_fast = mat_data['time_fast'].ravel()
pmt_pressure_df = pd.DataFrame({
    # 'timestamp_fast': pd.to_datetime(start_time) + pd.to_timedelta(time_fast, unit="s"),
    'time_fast'     : mat_data['time_fast'].ravel(),
    'PMT'      : mat_data['PMT_OH_1'].ravel(),
    'Cam_trig'      : mat_data['Cam_trig'].ravel(),
    'P1'            : mat_data['P1'].ravel(),
    'P2'            : mat_data['P2'].ravel(),
    'P3'            : mat_data['P3'].ravel(),
    'Pref'          : mat_data['Pref'].ravel(),
})


plot_pmt(pmt_pressure_df['PMT'])
# plot_massflows()