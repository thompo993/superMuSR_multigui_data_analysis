#=======================================================================
# amplitude_phs_analysis.py  — fixed
#=======================================================================
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
import warnings
import json
from pprint import pprint

warnings.filterwarnings("ignore")


#=======================================================================
# inputs
#=======================================================================
parent_dir = r"./test_parent_dir"
OUTPUT_DIR = r"./data"
N_CHANNELS = 8
NORMALISE  = True


#=======================================================================
# helper functions
#=======================================================================
def extract_daq_from_filename(filename):
    stem = Path(str(filename)).stem
    last_char = stem[-1]
    if last_char.isdigit() and 0 <= int(last_char) <= 4:
        return int(last_char)
    else:
        print("Digitiser ID outside of expected range (0-4). "
              "Please check parent directory.")


def group_files_by_digitiser(parent_directory):
    parent_directory = Path(parent_directory)
    groups, file_digitiser_map = {}, {}
    for file in parent_directory.iterdir():
        if not file.is_file():
            continue
        digitiser_id = extract_daq_from_filename(file.name)
        if digitiser_id is None:
            print(f"No digitiser ID found for: {file.name}")
            continue
        file_digitiser_map[file] = digitiser_id
        groups.setdefault(digitiser_id, []).append(file)
    return groups, file_digitiser_map


def extract_and_read_hk_from_groups(groups):
    hk_map, trigger_counts = {}, {}
    for digitiser_id, files in groups.items():
        hk_file = None
        for f in files:
            name = f.name.lower()
            if "hk" in name or f.suffix.lower() in (".set", ".json"):
                hk_file = f
                break
        hk_map[digitiser_id] = hk_file
        if hk_file is None:
            trigger_counts[digitiser_id] = None
            continue
        trigger_value = None
        if hk_file.suffix.lower() == ".json":
            try:
                with open(hk_file) as fp:
                    data = json.load(fp)
                def find_trigger(d):
                    for k, v in d.items():
                        if "trigger" in k.lower():
                            return v
                        if isinstance(v, dict):
                            r = find_trigger(v)
                            if r is not None:
                                return r
                    return None
                trigger_value = find_trigger(data)
            except Exception as e:
                print("JSON read error:", e)
        elif hk_file.suffix.lower() == ".set":
            try:
                with open(hk_file) as fp:
                    for line in fp:
                        if "=" in line:
                            k, v = line.split("=", 1)
                            if "trigger" in k.lower():
                                trigger_value = v.strip()
                                break
            except Exception as e:
                print("SET read error:", e)
        trigger_counts[digitiser_id] = trigger_value
    return hk_map, trigger_counts


def build_digitiser_summary(groups):
    hk_map, trigger_counts = extract_and_read_hk_from_groups(groups)
    summary_list = []
    seen_ids = set()                        # guard against duplicate entries
    for digitiser_id, files in groups.items():
        if digitiser_id in seen_ids:
            continue
        seen_ids.add(digitiser_id)
        time_hist_file = amp_hist_file = None
        for f in files:
            name = f.name.lower()
            if f.suffix.lower() == ".csv":
                if "time" in name:
                    time_hist_file = f
                elif "amp" in name or "amplitude" in name:
                    amp_hist_file = f
        summary_list.append({
            "digitiser_id":        digitiser_id,
            "hk_file":             hk_map.get(digitiser_id),
            "trigger_counter":     trigger_counts.get(digitiser_id),
            "time_histogram":      time_hist_file,
            "amplitude_histogram": amp_hist_file,
        })
    return summary_list


def load_amplitude_csv(filepath):
    """
    Load amplitude histogram CSV.

    Handles two formats:
      (a) Clean 8-column file  — col 0 = ch0 … col 7 = ch7
      (b) File saved with a pandas row-index (leading comma) — 9 columns,
          col 0 is the spurious integer index, col 1..8 are the channels.
          Detected by NaN in cell [0,0] AND a monotonically non-decreasing
          all-integer first column; dropped automatically.
    """
    df = pd.read_csv(filepath, header=None, encoding="utf-8")

    # Detect & drop spurious row-index column
    if df.shape[1] > 1 and pd.isna(df.iloc[0, 0]):
        candidate = df.iloc[1:, 0]
        floated   = pd.to_numeric(candidate, errors="coerce").dropna()
        if len(floated) > 0:
            is_monotone = bool((floated.diff().dropna() >= 0).all())
            is_integers = bool((floated % 1 == 0).all())
            if is_monotone and is_integers:
                df = df.iloc[:, 1:].reset_index(drop=True)

    # Drop first row if it is all-NaN or all-zero (header artefact)
    first_row = pd.to_numeric(df.iloc[0], errors="coerce")
    if first_row.isna().all() or (first_row.fillna(0) == 0).all():
        df = df.iloc[1:].reset_index(drop=True)

    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


#=======================================================================
# Gaussian fit
#=======================================================================
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


def fit_gaussian_bg_subtracted(x_win, y_raw_win, feat_x, est_width):
    """
    Fit a Gaussian to a peak without background subtraction.

    Uses raw counts directly, which is more robust for peaks sitting
    on a falling continuum.
    """
    if len(x_win) < 4:
        return None, None

    y_signal = y_raw_win.astype(float)
    valid = y_signal > 0
    if valid.sum() < 4:
        return None, None

    xv, yv = x_win[valid], y_signal[valid]
    x_lo, x_hi = float(xv[0]), float(xv[-1])

    try:
        p0 = [float(np.max(yv)), feat_x, max(float(est_width), 2.0)]
        popt, pcov = curve_fit(gaussian, xv, yv, p0=p0, maxfev=10000)
        mean = float(popt[1])
        if not (x_lo <= mean <= x_hi):
            return None, None
        var      = float(pcov[1, 1])
        mean_err = float(np.sqrt(var)) if var >= 0 else None
        return mean, mean_err
    except (RuntimeError, ValueError):
        return None, None


#=======================================================================
# Valley Gaussian fit
#=======================================================================
def fit_gaussian_valley(x_win, y_raw_win, feat_x, est_width):
    """Fit a Gaussian to the inverted signal around a valley (dip)."""
    if len(x_win) < 4:
        return None, None

    y_inv = -y_raw_win.astype(float)
    y_inv -= y_inv.min()

    valid = y_inv > 0
    if valid.sum() < 4:
        return None, None

    xv, yv = x_win[valid], y_inv[valid]
    x_lo, x_hi = float(xv[0]), float(xv[-1])

    try:
        p0 = [float(np.max(yv)), feat_x, max(float(est_width), 2.0)]
        popt, pcov = curve_fit(gaussian, xv, yv, p0=p0, maxfev=10000)
        mean = float(popt[1])
        if not (x_lo <= mean <= x_hi):
            return None, None
        var      = float(pcov[1, 1])
        mean_err = float(np.sqrt(var)) if var >= 0 else None
        return mean, mean_err
    except (RuntimeError, ValueError):
        return None, None


#=======================================================================
# Feature detection — single peak OR valley per channel
#=======================================================================
def _fit_features(x, y_smooth, y_norm, feature_type="peak",
                  window_frac=0.10, min_x=0.0):
    """
    Find the single most prominent peak (or valley) and fit a Gaussian.
    
    Peak detection strategy
    -----------------------
    1. Searches within range [300, n-95] to avoid electrical noise & tail
    2. Finds approximate peak/valley window using find_peaks
    3. Locates the maximum value within ±20% of data range around the window
    """
    n = len(x)
    if n < 5:
        return []

    # Define search range: from index 300 to (n - 95)
    search_start_idx = 300
    search_end_idx = max(search_start_idx + 1, n - 95)
    
    if search_end_idx <= search_start_idx:
        print(f"    [DEBUG {feature_type}] Search range invalid: start={search_start_idx}, end={search_end_idx}")
        return []
    
    # Slice the data to the search range
    x_search = x[search_start_idx:search_end_idx]
    y_smooth_search = y_smooth[search_start_idx:search_end_idx]
    y_norm_search = y_norm[search_start_idx:search_end_idx]
    
    print(f"    [DEBUG {feature_type}] Search range: indices {search_start_idx} to {search_end_idx}, x values {x_search[0]:.1f} to {x_search[-1]:.1f}")

    signal = -y_smooth_search if feature_type == "valley" else y_smooth_search

    sig_range  = float(signal.max() - signal.min())
    prominence = max(sig_range * 0.02, 1e-12)
    min_dist   = max(2, len(x_search) // 20)

    print(f"    [DEBUG {feature_type}] n={len(x_search)}, sig_range={sig_range:.2f}, prominence={prominence:.4f}, min_dist={min_dist}")
    print(f"    [DEBUG {feature_type}] signal min/max: {signal.min():.2f}/{signal.max():.2f}")

    # Find approximate peak/valley windows
    indices, props = find_peaks(
        signal,
        prominence=prominence,
        distance=min_dist,
        width=1,
        rel_height=0.5,
    )
    
    print(f"    [DEBUG {feature_type}] find_peaks returned {len(indices)} approximate features")
    
    if len(indices) == 0:
        print(f"    [DEBUG {feature_type}] No features found in search range")
        return []
    
    print(f"    [DEBUG {feature_type}] Approximate peak indices (local): {indices}")
    print(f"    [DEBUG {feature_type}] x values at approximate peaks: {x_search[indices]}")

    # For each approximate peak, find the max within ±20% of data range
    data_range = float(x_search[-1] - x_search[0])
    window_width = data_range * 0.20
    
    print(f"    [DEBUG {feature_type}] Data range: {data_range:.1f}, ±20% window width: {window_width:.1f}")

    best_idx = None
    best_signal_value = -np.inf
    
    for approx_idx in indices:
        approx_x = x_search[approx_idx]
        x_min = approx_x - window_width / 2.0
        x_max = approx_x + window_width / 2.0
        
        # Find max within this window
        window_mask = (x_search >= x_min) & (x_search <= x_max)
        if window_mask.sum() == 0:
            continue
        
        local_max_idx = np.argmax(signal[window_mask])
        local_max_global_idx = np.where(window_mask)[0][local_max_idx]
        local_max_value = signal[local_max_global_idx]
        
        print(f"    [DEBUG {feature_type}] Approx peak at x={approx_x:.1f}, window [{x_min:.1f}, {x_max:.1f}], max at x={x_search[local_max_global_idx]:.1f}, value={local_max_value:.2f}")
        
        if local_max_value > best_signal_value:
            best_signal_value = local_max_value
            best_idx = local_max_global_idx
    
    if best_idx is None:
        print(f"    [DEBUG {feature_type}] Failed to find max within windows")
        return []
    
    # Convert to global index
    best_global_idx = search_start_idx + best_idx
    est_width = max(5.0, window_width / 4.0)
    
    print(f"    [DEBUG {feature_type}] Final selected peak: local_idx={best_idx}, global_idx={best_global_idx}, x={x[best_global_idx]:.1f}, est_width={est_width:.2f}")

    feat_x = float(x[best_global_idx])

    # ------------------------------------------------------------------
    # Build fitting window
    # ------------------------------------------------------------------
    half_win = min(est_width * 4.0, 200.0)
    half_win = max(half_win, 5.0)

    mask = (x >= feat_x - half_win) & (x <= feat_x + half_win)
    x_win = x[mask]
    y_win = y_norm[mask]
    
    print(f"    [DEBUG {feature_type}] Fitting window: feat_x={feat_x:.1f}, half_win={half_win:.1f}, fit_points={mask.sum()}")

    # ------------------------------------------------------------------
    # Gaussian fit
    # ------------------------------------------------------------------
    if feature_type == "peak":
        mean, mean_err = fit_gaussian_bg_subtracted(x_win, y_win, feat_x, est_width)
    else:
        mean, mean_err = fit_gaussian_valley(x_win, y_win, feat_x, est_width)

    print(f"    [DEBUG {feature_type}] Fit result: mean={mean}, mean_err={mean_err}")

    result = {
        "feature_type":      feature_type,
        "feature_rank":      1,
        "bin_index":         int(best_global_idx),
        "smoothed_x":        feat_x,
        "smoothed_y":        float(y_smooth[best_global_idx]),
        "gaussian_mean":     mean,
        "gaussian_mean_err": mean_err,
        "n_fit_points":      int(mask.sum()),
        "fallback":          False,
    }
    return [result]


#=======================================================================
# Main analysis
#=======================================================================
def analyse_amplitude_phs(digitiser_summary,
                           n_channels    = N_CHANNELS,
                           normalise     = NORMALISE,
                           output_dir    = OUTPUT_DIR,
                           savgol_window = 11,
                           savgol_poly   = 3,
                           min_x         = 0.0):
    """
    For every digitiser × channel:
      1. Load amplitude histogram CSV (auto-detects spurious index column)
      2. Optionally normalise by trigger_counter
      3. Trim leading zero-bins; smooth with Savitzky-Golay
      4. Detect single peak and single valley; fit Gaussians.
         Channels where the photopeak is above the ADC range (pure Compton
         continuum) are now correctly reported as 'above_adc_range' instead
         of spuriously returning the first data bin.
      5. Write per-channel, per-digitiser, and master CSV files

    Output files (written to *output_dir*)
    ───────────────────────────────────────
    daq{N}_ch{C}_peaks.csv
    daq{N}_ch{C}_valleys.csv
    daq{N}_all_channels.csv
    all_digitisers_summary.csv

    CSV columns
    ───────────
    digitiser_id | channel | normalised | norm_factor | feature_type |
    feature_rank | bin_index | smoothed_x | smoothed_y | gaussian_mean |
    gaussian_mean_err | n_fit_points | fallback | fallback_reason
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_summary = sorted(digitiser_summary, key=lambda d: d["digitiser_id"])
    all_rows = []

    for entry in sorted_summary:
        dig_id          = entry["digitiser_id"]
        amp_file        = entry["amplitude_histogram"]
        trigger_counter = entry["trigger_counter"]

        # ── normalisation ────────────────────────────────────────────
        norm_factor  = 1.0
        norm_applied = False
        if normalise and trigger_counter is not None:
            try:
                nf = float(trigger_counter)
                if nf > 0:
                    norm_factor  = nf
                    norm_applied = True
            except (ValueError, TypeError):
                print(f"  [DAQ {dig_id}] Cannot convert trigger_counter "
                      f"'{trigger_counter}' to float — skipping normalisation.")

        # ── load CSV ─────────────────────────────────────────────────
        if amp_file is None:
            print(f"[DAQ {dig_id}] No amplitude CSV — skipping.")
            continue
        try:
            df = load_amplitude_csv(amp_file)
            print(f"[DAQ {dig_id}] Loaded {amp_file.name} "
                  f"({df.shape[0]} bins × {df.shape[1]} channels)")
        except Exception as exc:
            print(f"[DAQ {dig_id}] Failed to read amplitude CSV: {exc}")
            continue

        dig_rows = []

        for ch in range(n_channels):
            if ch >= df.shape[1]:
                print(f"  [DAQ {dig_id}] ch{ch}: not present — skipping.")
                continue

            raw    = df.iloc[:, ch].values.astype(float)
            normed = raw / norm_factor
            x_full = np.arange(len(normed), dtype=float)

            # Trim leading zeros
            nonzero = np.where(normed > 0)[0]
            if len(nonzero) == 0:
                print(f"  [DAQ {dig_id}] ch{ch}: all zeros — skipping.")
                continue
            start = int(nonzero[0])

            x_a      = x_full[start:]
            normed_a = normed[start:]

            # Savitzky-Golay smooth
            win = min(savgol_window, len(normed_a) - 1)
            if win % 2 == 0:
                win -= 1
            win      = max(win, savgol_poly + 2)
            smooth_a = savgol_filter(normed_a, window_length=win,
                                     polyorder=savgol_poly)

            # Detect & fit
            peaks   = _fit_features(x_a, smooth_a, normed_a, "peak",   min_x=min_x)
            valleys = _fit_features(x_a, smooth_a, normed_a, "valley", min_x=min_x)

            # Remap trimmed-array coordinates → full-array coordinates
            # (skipped for sentinel rows where bin_index is None)
            for feat_list in (peaks, valleys):
                for feat in feat_list:
                    if feat.get("bin_index") is not None:
                        feat["bin_index"]  = int(feat["bin_index"]  + start)
                        feat["smoothed_x"] = float(feat["smoothed_x"] + start)
                    if feat.get("gaussian_mean") is not None:
                        feat["gaussian_mean"] = float(feat["gaussian_mean"] + start)

            # Assemble rows
            base    = {"digitiser_id": dig_id, "channel": ch,
                       "normalised": norm_applied, "norm_factor": norm_factor}
            ch_rows = [{**base, **f} for f in peaks + valleys]
            dig_rows.extend(ch_rows)
            all_rows.extend(ch_rows)

            # Per-channel CSVs
            pk_df = pd.DataFrame([r for r in ch_rows if r["feature_type"] == "peak"])
            vl_df = pd.DataFrame([r for r in ch_rows if r["feature_type"] == "valley"])
            pk_path = output_dir / f"daq{dig_id}_ch{ch}_peaks.csv"
            vl_path = output_dir / f"daq{dig_id}_ch{ch}_valleys.csv"
            pk_df.to_csv(pk_path, index=False)
            vl_df.to_csv(vl_path, index=False)

            # Diagnostic print
            pk_info = ""
            if len(pk_df):
                r         = pk_df.iloc[0]
                fb_reason = r.get("fallback_reason", "")
                if fb_reason == "above_adc_range":
                    pk_info = "NO PEAK (photopeak above ADC range)"
                else:
                    gm = (f"{r['gaussian_mean']:.1f}"
                          if pd.notna(r.get("gaussian_mean")) else "no fit")
                    fb = " [fallback]" if r.get("fallback") else ""
                    pk_info = f"peak@{r['bin_index']}(gaus={gm}){fb}"

            vl_info = ""
            if len(vl_df):
                r    = vl_df.iloc[0]
                gm   = (f"{r['gaussian_mean']:.1f}"
                        if pd.notna(r.get("gaussian_mean")) else "no fit")
                vl_info = f"valley@{r['bin_index']}(gaus={gm})"

            print(f"  ch{ch}: {pk_info or 'NO PEAK'}, {vl_info or 'no valley'}")

        if dig_rows:
            dig_path = output_dir / f"daq{dig_id}_all_channels.csv"
            pd.DataFrame(dig_rows).to_csv(dig_path, index=False)
            print(f"[DAQ {dig_id}] Combined CSV → {dig_path.name}")

    if all_rows:
        master_path = output_dir / "all_digitisers_summary.csv"
        pd.DataFrame(all_rows).to_csv(master_path, index=False)
        print(f"\nMaster summary → {master_path}")

    print("\nAnalysis complete.")


#=======================================================================
# workflow
#=======================================================================
groups, file_digitiser_map = group_files_by_digitiser(parent_dir)

pprint({k: [p.name for p in v] for k, v in groups.items()})

print("\nCounts per digitiser:")
for digitiser_id, files in groups.items():
    print(f"  Digitiser {digitiser_id}: {len(files)} files")

digitiser_summary = build_digitiser_summary(groups)

analyse_amplitude_phs(
    digitiser_summary,
    n_channels    = N_CHANNELS,
    normalise     = NORMALISE,
    output_dir    = OUTPUT_DIR,
    savgol_window = 11,
    savgol_poly   = 3,
    min_x         = 0.0,
)