import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from nptdms import TdmsFile
import scipy.io as sio
from datetime import datetime, timedelta


crossing_threshold = 0

def calculate_U_ER(pmt_pressure_df: pd.DataFrame, flow_df: pd.DataFrame):
    cross_idx_pmt = np.where(pmt_pressure_df['PMT'] <= crossing_threshold)[0]

    if cross_idx_pmt.size == 0:
        # No crossing found — skip cleanly
        msg = (f"[WARN] No PMT crossing at threshold {crossing_threshold}. "
            f"Signal min={np.min(pmt_pressure_df['PMT']):.3g}, max={np.max(pmt_pressure_df['PMT']):.3g}")
        print(msg)
        return None, None, None

    i_cross_pmt = int(cross_idx_pmt[0])
    cross_pmt_time_value = pmt_pressure_df['timestamps'][i_cross_pmt]
    print("First near-zero crossing index for pmt:", i_cross_pmt)

    nearest_idx_flow = (flow_df['Time'] - cross_pmt_time_value).abs().idxmin()
    cross_flow_time_value = flow_df.loc[nearest_idx_flow, 'Time']
    print("Nearest near-zero crossing index for flow:", nearest_idx_flow)


    cross_time_diff = abs(cross_flow_time_value - cross_pmt_time_value)
    print('Blow off time pmt:',cross_pmt_time_value)
    print('Blow off time flow:',cross_flow_time_value)
    print('Blow off time difference is:',cross_time_diff)

    # peak_idx_pmt = np.where((pmt_pressure_df['PMT'][:1] - pmt_pressure_df['PMT'][:-10]) > 0.1)[0]
    # i_peak_pmt = int(peak_idx_pmt[0])
    # print("First peak index:", i_peak_pmt)

    area_cross_section = 1.51e-4 #m^3
    pressure = 1e5 #pascal
    temperature = 273.5 #K
    R_molar = 8.314 # J/(mol K)

    air_volumflow_blow_off = flow_df['air_volum_flow'][nearest_idx_flow]
    print(f'Air at blow off{air_volumflow_blow_off}')
    CH4_volumflow_blow_off = flow_df['CH4_volum_flow'][nearest_idx_flow]
    print(f'CH4 at blow off{CH4_volumflow_blow_off}')
    total_volumflow_blow_off = air_volumflow_blow_off + CH4_volumflow_blow_off
    U_blow_off = float(total_volumflow_blow_off / area_cross_section / 1000 / 60)

    air_mole_blow_off = (pressure*air_volumflow_blow_off)/(R_molar*temperature)
    CH4_mole_blow_off = (pressure*CH4_volumflow_blow_off)/(R_molar*temperature)

    ER_blow_off = float((9.5/1) / (air_mole_blow_off/CH4_mole_blow_off))
    print(f'ER: {ER_blow_off} U: {U_blow_off}')

    return ER_blow_off, U_blow_off, cross_time_diff

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

# Start a looop
# Here it should read freq_results.csv and find the tdms file to read

flow_dataFrame = load_tdms_data(tdms)
pmt_pressure_dataFrame = load_mat_data(mat)

er, u , time_diff = calculate_U_ER(pmt_pressure_dataFrame, flow_dataFrame)

# append er, u , time_diff to freq_results.csv


