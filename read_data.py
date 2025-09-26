import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import re

from datetime import datetime, timedelta
from pathlib import Path
from nptdms import TdmsFile

def iter_data_files(base_path, include_subfolders=False):
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

def pair_mat_tdms(files, *, tolerance_seconds=20, group_by_dir=True):
    """
    Pair .mat with the closest-in-time .tdms, within tolerance.
    If group_by_dir=True, only pair files that share the same parent directory.
    Returns (pairs, unmatched_mats, unmatched_tdms).
    """
    from collections import defaultdict

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

def find_channel_names():
    print([g.name for g in tdms.groups()])
    for g in tdms.groups():
        print(g.name, [c.name for c in g.channels()])

def plot_massflows():
    # ----- plot (single subplot/axes) -----
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))  # using the subplot API as requested
    flow_df.plot(ax=ax, x="Time", y=["air_volum_flow", "CH4_volum_flow"], linewidth=1)

    ax.set_title("Mass Flow vs Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mass flow")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    plt.show()

def plot_pmt(show_plot):
    # Find first index where we cross from >thr to <=thr
    cross_idx_pmt = np.where(pmt_pressure_df['PMT'] <= crossing_threshold)[0]
    i_cross_pmt = int(cross_idx_pmt[0])
    print("First near-zero crossing index:", i_cross_pmt)

    nearest_idx_flow = (flow_df['Time'] - pmt_pressure_df['timestamps'][i_cross_pmt]).abs().idxmin()
    cross_flow_time_value = flow_df.loc[nearest_idx_flow, 'Time']
    print("Nearest near-zero crossing index for flow:", nearest_idx_flow, cross_flow_time_value)

    # peak_idx_pmt = np.where((pmt_pressure_df['PMT'][:1] - pmt_pressure_df['PMT'][:-10]) > 0.1)[0]
    # i_peak_pmt = int(peak_idx_pmt[0])
    # print("First peak index:", i_peak_pmt)


    if show_plot:

        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(11, 4))
        pmt_pressure_df.plot(ax=ax1, x='timestamps', y='PMT', linewidth=1)
        ax1.axvline(pmt_pressure_df['timestamps'][i_cross_pmt], color='red')
        # ax1.axvline(pmt_pressure_df['timestamps'][i_peak_pmt], color='red')
        ax1.set_title("PMT vs Time")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("PMT")
        ax1.grid(True, which="both", linestyle="--", alpha=0.4)
        ax1.legend()

        flow_df.plot(ax=ax2, x="Time", y=["air_volum_flow", "CH4_volum_flow"], linewidth=1)
        ax2.axvline(cross_flow_time_value, color='red')
        ax2.set_title("Mass Flow vs Time")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Mass flow")
        ax2.grid(True, which="both", linestyle="--", alpha=0.4)
        ax2.legend()

        fig.tight_layout()
        ax2.set_xlim(ax1.get_xlim())
        plt.show()

def calculate_U_ER():
    cross_idx_pmt = np.where(pmt_pressure_df['PMT'] <= crossing_threshold)[0]
    i_cross_pmt = int(cross_idx_pmt[0])
    print("First near-zero crossing index:", i_cross_pmt)

    nearest_idx_flow = (flow_df['Time'] - pmt_pressure_df['timestamps'][i_cross_pmt]).abs().idxmin()
    print("Nearest near-zero crossing index for flow:", nearest_idx_flow)

    # peak_idx_flow

    area_cross_section = 1.51e-4 #m^3
    pressure = 1e5 #pascal
    temperature = 273.5 #K
    R_molar = 8.314 # J/(mol K)

    air_volumflow_blow_off = flow_df['air_volum_flow'][nearest_idx_flow]
    print(air_volumflow_blow_off)
    CH4_volumflow_blow_off = flow_df['CH4_volum_flow'][nearest_idx_flow]
    print(CH4_volumflow_blow_off)
    total_volumflow_blow_off = air_volumflow_blow_off + CH4_volumflow_blow_off
    U_blow_off = total_volumflow_blow_off / area_cross_section / 1000 / 60

    air_mole_blow_off = (pressure*air_volumflow_blow_off)/(R_molar*temperature)
    CH4_mole_blow_off = (pressure*CH4_volumflow_blow_off)/(R_molar*temperature)

    ER_blow_off = (9.5/1) / (air_mole_blow_off/CH4_mole_blow_off)
    print(f'ER: {ER_blow_off} U: {U_blow_off}')

    # fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    # ax.plot([ER_blow_off],[U_blow_off], 'o-' , color = 'red')
    # ax.set_xlabel(r'$\phi$ [-]')
    # ax.set_ylabel('Velocity [m/s]')
    # plt.show()

def load_tdms_data(tdms_path):
    #load the tdms data
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
                'air_volum_flow' : air_volum_flow,
                'CH4_volum_flow' : CH4_volum_flow
        })

        return flow_df

def load_mat_data(mat_path):
    #load the mat data

    m = sio.loadmat(mat_path, squeeze_me=True, simplify_cells=True)
    mat_data = m['data']            # now a plain dict (SciPy ≥1.7)
    # print(list(mat_data.keys()))

    posix = float(mat_data['timestamp_fast_posix'])
    start_time_pmt = datetime(1970,1,1) + timedelta(seconds=posix)
    t_rel = np.asarray(mat_data['time_fast'], dtype=float).ravel()  # seconds
    timestamps = pd.to_datetime(start_time_pmt, utc=True) + pd.to_timedelta(t_rel, unit='s') - pd.Timedelta(hours=1)
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
# Choose a threshold for "near zero" (1% of max is a decent default)
crossing_threshold = 0.1

base_path = Path(r'D:\202508Experiment_data_logging\test_mappe_2')
files = iter_data_files(base_path, include_subfolders=False)

pairs, um_mats, um_tdms = pair_mat_tdms(
    files,
    tolerance_seconds=20,   # tweak if needed
    group_by_dir=True
)

for mat, tdms in pairs:
    print("PAIR:", mat.name, "<->", tdms.name)
    flow_df = load_tdms_data(tdms)
    pmt_pressure_df = load_mat_data(mat)
    plot_pmt(True)
    calculate_U_ER()




if um_mats:
    print("\nUnmatched MAT:")
    for p in um_mats:
        print("  ", p.name)
if um_tdms:
    print("\nUnmatched TDMS:")
    for p in um_tdms:
        print("  ", p.name)


# for path in iter_data_files(base_path, False):
#     if path.suffix.lower() == ".tdms":
#         # process_tdms(path)
#         print("TDMS:", path)
#         # flow_df = load_tdms_data(path)
#     elif path.suffix.lower() == ".mat":
#         print("MAT :", path)
#         # pmt_pressure_df = load_mat_data(path)

#-------------------------------------------------------------------------------------------
# base_path = Path(r'G:\202508Experiment_data_logging\03_09_D_88mm_350mm')
# tdms_path = base_path / 'ER1_0,65_Log2_03.09.2025_08.46.16.tdms'
# mat_time_path = base_path / 'LBO_Sweep_2_8_46_19_posix.mat'
# mat_path = base_path / 'LBO_Sweep_2_8_46_19.mat'

# plot_U_ER()
# plot_pmt(True)
# plot_massflows()
