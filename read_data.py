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

_LOG_RE = re.compile(r'(?:^|[_\-\s])log\s*([0-9]+)(?=[_\-\s.]|$)', re.IGNORECASE)
def get_log_no(filename: str) -> int | None:
    """Extract the log number from a filename (stem or full name)."""
    m = _LOG_RE.search(filename)
    return int(m.group(1)) if m else None

# ER like ER1_0,65 → 0.65 (comma decimal)
_ER_RE = re.compile(r'ER[0-9]+_([0-9],[0-9]+)', re.IGNORECASE)
def get_er_est(name: str) -> float | None:
    m = _ER_RE.search(name)
    return float(m.group(1).replace(',', '.')) if m else None

def find_channel_names(tdms: Path):
    print([g.name for g in tdms.groups()])
    for g in tdms.groups():
        print(g.name, [c.name for c in g.channels()])

def plot_massflows(flow_df: pd.DataFrame):
    """
        ----- plot (single subplot/axes) -----    
    """
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))  # using the subplot API as requested
    flow_df.plot(ax=ax, x="Time", y=["air_volum_flow", "CH4_volum_flow"], linewidth=1)

    ax.set_title("Mass Flow vs Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mass flow")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    plt.show()

def plot_pmt(pmt_pressure_df: pd.DataFrame, flow_df: pd.DataFrame,show_plot: bool):
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

def calculate_U_ER(pmt_pressure_df: pd.DataFrame, flow_df: pd.DataFrame, show_plot = False):
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
    print(f'Air at blow off{CH4_volumflow_blow_off}')
    total_volumflow_blow_off = air_volumflow_blow_off + CH4_volumflow_blow_off
    U_blow_off = float(total_volumflow_blow_off / area_cross_section / 1000 / 60)

    air_mole_blow_off = (pressure*air_volumflow_blow_off)/(R_molar*temperature)
    CH4_mole_blow_off = (pressure*CH4_volumflow_blow_off)/(R_molar*temperature)

    ER_blow_off = float((9.5/1) / (air_mole_blow_off/CH4_mole_blow_off))
    print(f'ER: {ER_blow_off} U: {U_blow_off}')

    if show_plot:
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(11, 4))
        # pmt_pressure_df.plot(ax=ax1, x='timestamps', y=['PMT', 'P1'], linewidth=1)
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

    return ER_blow_off, U_blow_off, cross_time_diff
    # fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    # ax.plot([ER_blow_off],[U_blow_off], 'o-' , color = 'red')
    # ax.set_xlabel(r'$\phi$ [-]')
    # ax.set_ylabel('Velocity [m/s]')
    # plt.show()

def detect_pmt_peaks(df, col='PMT',
                     smooth_ms=10,         # small smoothing for noise
                     baseline_ms=300,      # rolling-median baseline removal
                     min_distance_s=0.30,  # refractory time between peaks
                     min_width_ms=10,      # discard ultra-narrow blips
                     prominence_sigma=10.0, # how strong above noise
                     rel_height=0.5):      # width at 50% prominence
    """
    df: DataFrame with columns ['timestamps', col]
    Returns: peaks_df with timestamp, height, prominence, width_s, left_base_ts, right_base_ts
    """
    ts = pd.to_datetime(df['timestamps'])
    x  = df[col].to_numpy(dtype=float)

    # --- sampling rate ---
    dt = np.median(np.diff(ts.view('int64'))) / 1e9  # seconds
    fs = 1.0 / dt

    # --- baseline remove with rolling median (robust to outliers) ---
    win_baseline = max(3, int(round(baseline_ms/1000 * fs)))
    baseline = pd.Series(x).rolling(win_baseline, center=True, min_periods=1).median().to_numpy()
    y = x - baseline

    # --- light smoothing to tame high-freq noise (keeps peak timing) ---
    if smooth_ms and smooth_ms > 0:
        w = max(3, int(round(smooth_ms/1000 * fs)) | 1)  # odd
        y = savgol_filter(y, window_length=w, polyorder=2, mode='interp')

    # --- robust noise estimate (MAD) to set data-driven prominence ---
    med = np.median(y)
    sigma = 1.4826 * np.median(np.abs(y - med))
    min_prom = max(1e-3, prominence_sigma * sigma)

    distance = int(round(min_distance_s * fs))
    min_width = max(1, int(round(min_width_ms/1000 * fs)))

    idx, props = find_peaks(
        y,
        prominence=min_prom,
        distance=distance,
        width=min_width,
        rel_height=rel_height
    )

    # Optional: refine each peak to the local maximum in the raw signal
    # (in case smoothing shifted it slightly)
    if idx.size:
        halfw = max(1, int(0.02 * fs))  # search ±20 ms
        ref = []
        for i in idx:
            lo = max(0, i-halfw); hi = min(len(x), i+halfw+1)
            i_ref = lo + np.argmax(x[lo:hi])
            ref.append(i_ref)
        idx = np.array(ref, dtype=int)

    # Build result table
    widths_s = props['widths'] * dt
    left_b   = props['left_bases'].astype(int, copy=False)
    right_b  = props['right_bases'].astype(int, copy=False)

    peaks_df = pd.DataFrame({
        'idx': idx,
        'timestamp': ts.to_numpy()[idx],
        'height': x[idx],
        'prominence': props['prominences'],
        'width_s': widths_s,
        'left_base_ts': ts.to_numpy()[left_b],
        'right_base_ts': ts.to_numpy()[right_b],
    })

    return peaks_df

def peak_period_frequency(peaks_df, timestamp_col='timestamp'):
    # sort just in case
    ts = pd.to_datetime(peaks_df[timestamp_col]).sort_values().reset_index(drop=True)
    if len(ts) < 2:
        raise ValueError("Need at least two peaks to compute intervals/frequency.")

    # periods (s) between peaks
    periods_s = ts.diff().dt.total_seconds().iloc[1:].to_numpy()

    # instantaneous frequencies (Hz)
    freq_inst = 1.0 / periods_s

    # summary stats
    period_mean = periods_s.mean()
    period_std  = periods_s.std(ddof=1) if len(periods_s) > 1 else 0.0     # sample std

    # Preferred overall frequency estimate: invert mean period (less biased)
    freq_mean = 1.0 / period_mean
    # Also report the mean of instantaneous frequencies (sometimes people want this)
    freq_mean_inst = freq_inst.mean()
    # Std of instantaneous frequency
    freq_std = freq_inst.std(ddof=1) if len(freq_inst) > 1 else 0.0

    # Robust (median / MAD) in case of outliers
    period_med = np.median(periods_s)
    period_mad = 1.4826 * np.median(np.abs(periods_s - period_med))
    freq_med   = np.median(freq_inst)
    freq_mad   = 1.4826 * np.median(np.abs(freq_inst - freq_med))

    return {
        "n_peaks": len(ts),
        "n_intervals": len(periods_s),
        "periods_s": periods_s,           # array
        "freq_inst_Hz": freq_inst,        # array
        "period_mean_s": float(period_mean),
        "period_std_s": float(period_std),
        "period_median_s": float(period_med),
        "period_MAD_s": float(period_mad),
        "freq_mean_Hz": float(freq_mean),             # recommended point estimate
        "freq_mean_inst_Hz": float(freq_mean_inst),   # mean of 1/period
        "freq_std_Hz": float(freq_std),
        "freq_median_Hz": float(freq_med),
        "freq_MAD_Hz": float(freq_mad),
    }

def plot_with_peaks(pmt_pressure_df: pd.DataFrame,peaks_df: pd.DataFrame, matFileName: str, tdmsFileName: str):
    fig, ax = plt.subplots(1, 1, figsize=(11, 3.5))
    ax.plot(pmt_pressure_df['timestamps'], pmt_pressure_df['PMT'], label='PMT', linewidth=1)
    ax.scatter(peaks_df['timestamp'], peaks_df['height'], marker='o', color='red', s=18, zorder=3, label='Detected peaks')
    ax.set_title("PMT vs Time (with peaks)")
    ax.set_xlabel("Time")
    ax.set_ylabel("PMT")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fr'C:\Users\marha\Documents\Skole\master_project\pictures\{matFileName} and  {tdmsFileName}.png')
    plt.close()
    return

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

csv_rows: List[Dict] = []
def record_pair(file_mat: Path, file_tdms: Path, ER: float, velocity: float,time_diff):
    tdms_stem = file_tdms.stem
    csv_rows.append({
        "folder": file_mat.parent.name,                 # folder name of file A
        "mat_file": file_mat.name,                          # file A name
        "tdms_file": file_tdms.name,                   # file B name (the pairing)
        "pairing": f"{file_mat.stem} <> {tdms_stem}", # human-readable pairing label
        "log": get_log_no(tdms_stem),
        "er_est": get_er_est(tdms_stem),
        "ER": float(ER),
        "velocity": float(velocity),
        "time_diff" : time_diff
    })

# Choose a threshold for "near zero" (1% of max is a decent default)
crossing_threshold = 0
def main(do_LBO = False, do_Freq = False ):

    base_path = Path(r'data')
    files = iter_data_files(base_path, True)

    #Find the pairs in the code
    pairs, um_mats, um_tdms = pair_mat_tdms(
        files,
        tolerance_seconds=55,   # tweak if needed
        group_by_dir=True
    )
    
    #To track unused files
    total_files = len(pairs)
    no_zero_cross = 0
    freq_fail     = 0
    freq_rows     = []


    #Main code
    for file_idx, (mat, tdms) in enumerate(pairs):
        log_no = get_log_no(tdms.stem)
        print(f'Files processed {file_idx+1}/{total_files}')
        print("PAIR:", mat.name, "<->", tdms.name)

        # ------- LBO path: logs 1–3 -------
        if do_LBO and log_no in {1,2,3}:
            print('LBO candidate (log 1-3)')
            flow_data_df = load_tdms_data(tdms)
            pmt_pressure_data_df = load_mat_data(mat)
            
            result = calculate_U_ER(pmt_pressure_data_df, flow_data_df)
            if result == (None, None, None):
                no_zero_cross += 1
                #continue to next file
                continue
            
            ER_pair, U_pair, time_difference = result
            record_pair(mat, tdms, ER_pair, U_pair, time_difference)
        
        # ------- Frequency path: logs 4–6 -------
        elif do_Freq and log_no in {4}:
            print('Frequency candidate (log 4-6)')
            pmt_pressure_data_df = load_mat_data(mat)

            try:
                print('Detecting peaks')
                peaks = detect_pmt_peaks(pmt_pressure_data_df)
                print('Plotting')
                plot_with_peaks(pmt_pressure_data_df, peaks, mat.stem, tdms.stem)
                print('Calculating freq')
                stats = peak_period_frequency(peaks)
            except ValueError:
                freq_fail += 1
                print("Not enough peaks to compute frequency; skipping.")
                continue
            except Exception as e:
                freq_fail += 1
                print(f"Peak/frequency computation failed: {e}")
                continue

            print(f"Freq = {stats['freq_mean_Hz']:.3f} ± {stats['freq_std_Hz']:.3f} Hz "
                  f"(median {stats['freq_median_Hz']:.3f}, MAD {stats['freq_MAD_Hz']:.3f})")

            # collect for CSV
            freq_rows.append({
                "folder": mat.parent.name,
                "mat_file": mat.name,
                "tdms_file": tdms.name,
                "log_no": log_no,
                "n_peaks": stats["n_peaks"],
                "n_intervals": stats["n_intervals"],
                "freq_mean_Hz": stats["freq_mean_Hz"],
                "freq_std_Hz": stats["freq_std_Hz"],
                "freq_median_Hz": stats["freq_median_Hz"],
                "freq_MAD_Hz": stats["freq_MAD_Hz"],
            })


    # ------- Count and present unpaired files -------
    unpaired = 0
    if um_mats:
        print("\nUnmatched MAT:")
        for p in um_mats:
            print("  ", p.name, p.parent)
            unpaired += 1
    if um_tdms:
        print("\nUnmatched TDMS:")
        for p in um_tdms:
            print("  ", p.name, p.parent)
            unpaired += 1

    # ---- After the loop finishes ----
    if do_LBO and csv_rows:    
        script_dir = Path(__file__).resolve().parent
        out_csv = script_dir / "post_process_data.csv"
        csv_df = pd.DataFrame(csv_rows, columns=["folder","mat_file","tdms_file","pairing","time_diff","log","er_est","ER","velocity"])
        csv_df.to_csv(out_csv, index=False)
        print(f"Saved {len(csv_df)} rows to {out_csv}")
        print(f'No zero crossing: {no_zero_cross} and unpaired: {unpaired}')

    if do_Freq and freq_rows:
        out = Path("freq_results.csv")
        write_header = not out.exists()
        pd.DataFrame(freq_rows).to_csv(
            out, index=False,
            mode='a' if not write_header else 'w',
            header=write_header
        )
        print(f"Saved {len(freq_rows)} frequency rows to {out}")


    print(f'No-zero-cross (LBO) skipped: {no_zero_cross}')
    print(f'Frequency failures (not enough peaks): {freq_fail}')
    print(f'Unpaired files: {unpaired}')


main(False, True)






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

# base_path = Path(r'data\03_09_D_88mm_350mm')
# mat_path  = base_path / 'Up_8_ERp_0.65_PH2p_0_8_59_1.mat'
# pmt_pressure_peak_df = load_mat_data(mat_path)


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
