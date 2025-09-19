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

base_path = Path(r'D:\202508Experiment_data_logging\03_09_D_88mm_350mm')
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

df = pd.DataFrame({
    # 'timestamp_fast': timestamp,
    #'time_fast'     : np.asarray(getattr(mat_data, 'time_fast'), dtype=float).ravel(),
    'PMT_OH_1'      : np.asarray(getattr(mat_data, 'PMT_OH_1'), dtype=float).ravel(),
    'Cam_trig'      : np.asarray(getattr(mat_data, 'Cam_trig'), dtype=float).ravel(),
    'P1'            : np.asarray(getattr(mat_data, 'P1'), dtype=float).ravel(),
    'P2'            : np.asarray(getattr(mat_data, 'P2'), dtype=float).ravel(),
    'P3'            : np.asarray(getattr(mat_data, 'P3'), dtype=float).ravel(),
    'Pref'          : np.asarray(getattr(mat_data, 'Pref'), dtype=float).ravel(),
}).sort_values('timestamp_fast').reset_index(drop=True)

#Ikke helt sikker, men ser ut som at den tror hver verdi er en float og ikke en array med floats