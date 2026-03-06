#=======================================================================
# amplitude_phs_analysis.py
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
    for digitiser_id, files in groups.items():
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
# Gaussian fit with linear background subtraction
#=======================================================================
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


def fit_gaussian_bg_subtracted(x_win, y_raw_win, feat_x, est_width):
    """
    Fit a Gaussian to a peak after subtracting a linear background.

    The background is estimated by drawing a straight line between the
    left and right edges of the fitting window.  The Gaussian is then
    fitted to the background-subtracted raw counts.

    Parameters
    ----------
    x_win     : 1-D array — x values inside the fitting window
    y_raw_win : 1-D array — raw (or normalised) counts inside the window
    feat_x    : float     — initial guess for the peak centre
    est_width : float     — estimated half-width (e.g. from find_peaks widths)

    Returns
    -------
    (mean, mean_err) or (None, None) on failure.
    """
    if len(x_win) < 4:
        return None, None

    # Linear background from window edges
    bg_slope     = (float(y_raw_win[-1]) - float(y_raw_win[0])) / \
                   max(float(x_win[-1] - x_win[0]), 1.0)
    bg_intercept = float(y_raw_win[0]) - bg_slope * float(x_win[0])
    background   = bg_slope * x_win + bg_intercept

    y_signal = y_raw_win.astype(float) - background
    y_signal = np.maximum(y_signal, 0.0)

    valid = y_signal > 0
    if valid.sum() < 4:
        return None, None

    xv, yv = x_win[valid], y_signal[valid]
    x_lo, x_hi = float(xv[0]), float(xv[-1])

    try:
        p0 = [float(np.max(yv)), feat_x, max(float(est_width), 2.0)]
        popt, pcov = curve_fit(gaussian, xv, yv, p0=p0, maxfev=10000)
        mean = float(popt[1])
        # Reject unphysical fits
        if not (x_lo <= mean <= x_hi):
            return None, None
        var      = float(pcov[1, 1])
        mean_err = float(np.sqrt(var)) if var >= 0 else None
        return mean, mean_err
    except (RuntimeError, ValueError):
        return None, None


#=======================================================================
# Valley Gaussian fit (no background subtraction — valley is a dip)
#=======================================================================
def fit_gaussian_valley(x_win, y_raw_win, feat_x, est_width):
    """
    Fit a Gaussian to the inverted signal around a valley (dip).
    Uses the background-subtracted inverted counts.
    """
    if len(x_win) < 4:
        return None, None

    # Invert: turn the dip into a bump
    y_inv = -y_raw_win.astype(float)
    y_inv -= y_inv.min()           # shift so minimum is zero

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

    Peak fitting
    ------------
    Uses a linear background subtraction before fitting so that peaks
    sitting on a steeply falling Compton continuum are located accurately.
    The fitting window half-width is set to 4× the half-prominence width
    returned by find_peaks (physics-driven), capped at 200 bins.

    If no interior peak is found by find_peaks (e.g. the photopeak is
    below the ADC threshold and only its falling edge is visible), the
    global maximum of the smoothed spectrum is used as the peak location.

    Valley robustness
    -----------------
    A valley is only accepted if its smoothed count exceeds 0.5 % of the
    spectrum maximum, preventing the noise floor at the far end of the
    array from being mis-labelled as a valley.

    Returns
    -------
    list with at most one feature dict, or [] if none found.
    """
    n = len(x)
    if n < 5:
        return []

    signal = -y_smooth if feature_type == "valley" else y_smooth

    sig_range  = float(signal.max() - signal.min())
    prominence = max(sig_range * 0.02, 1e-12)
    min_dist   = max(2, n // 20)

    indices, props = find_peaks(
        signal,
        prominence=prominence,
        distance=min_dist,
        width=1,          # always request width so props['widths'] exists
        rel_height=0.5,
    )

    # Drop features at or below min_x
    if len(indices):
        keep = x[indices] > min_x
        indices = indices[keep]
        for key in props:
            if hasattr(props[key], '__len__') and len(props[key]) == len(keep):
                props[key] = props[key][keep]

    # Valley: reject noise-floor hits
    if feature_type == "valley" and len(indices):
        floor_threshold = float(y_smooth.max()) * 0.005
        keep = y_smooth[indices] > floor_threshold
        indices = indices[keep]
        for key in props:
            if hasattr(props[key], '__len__') and len(props[key]) == len(keep):
                props[key] = props[key][keep]

    # ------------------------------------------------------------------
    # Peak fallback: if find_peaks found nothing, use the global maximum.
    # This handles spectra where the photopeak is partially or fully below
    # the ADC threshold (only the falling right-hand edge is visible).
    # ------------------------------------------------------------------
    fallback_peak = False
    if feature_type == "peak" and len(indices) == 0:
        best_idx     = int(np.argmax(signal))
        est_width    = max(float(n) * 0.03, 5.0)   # rough 3 % of spectrum
        fallback_peak = True
    else:
        if len(indices) == 0:
            return []
        # Select the most prominent feature
        best_idx  = int(indices[np.argmax(signal[indices])])
        est_width = float(props["widths"][np.argmax(signal[indices])])

    feat_x = float(x[best_idx])

    # ------------------------------------------------------------------
    # Build fitting window
    #   • peaks   : 4× half-prominence width, capped at 200 bins
    #   • valleys : 4× width, capped at 200 bins
    # For fallback peaks (edge maximum) use right-side only.
    # ------------------------------------------------------------------
    half_win = min(est_width * 4.0, 200.0)
    half_win = max(half_win, 5.0)

    if fallback_peak:
        # Only right-side window available
        mask = (x >= feat_x) & (x <= feat_x + half_win)
    else:
        mask = (x >= feat_x - half_win) & (x <= feat_x + half_win)

    x_win = x[mask]
    y_win = y_norm[mask]

    # ------------------------------------------------------------------
    # Gaussian fit
    # ------------------------------------------------------------------
    if feature_type == "peak":
        mean, mean_err = fit_gaussian_bg_subtracted(x_win, y_win, feat_x, est_width)
    else:
        mean, mean_err = fit_gaussian_valley(x_win, y_win, feat_x, est_width)

    return [{
        "feature_type":      feature_type,
        "feature_rank":      1,
        "bin_index":         int(best_idx),
        "smoothed_x":        feat_x,
        "smoothed_y":        float(y_smooth[best_idx]),
        "gaussian_mean":     mean,
        "gaussian_mean_err": mean_err,
        "n_fit_points":      int(mask.sum()),
        "fallback":          fallback_peak if feature_type == "peak" else False,
    }]


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
      4. Detect single peak and single valley; fit Gaussians with
         linear background subtraction for accurate centre estimation
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
    gaussian_mean_err | n_fit_points | fallback
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
            for feat_list in (peaks, valleys):
                for feat in feat_list:
                    feat["bin_index"]  = int(feat["bin_index"]  + start)
                    feat["smoothed_x"] = float(feat["smoothed_x"] + start)
                    if feat["gaussian_mean"] is not None:
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
                r    = pk_df.iloc[0]
                gm   = f"{r['gaussian_mean']:.1f}" if pd.notna(r.get('gaussian_mean')) else "no fit"
                fb   = " [fallback]" if r.get('fallback') else ""
                pk_info = f"peak@{r['bin_index']}(gaus={gm}){fb}"
            vl_info = ""
            if len(vl_df):
                r    = vl_df.iloc[0]
                gm   = f"{r['gaussian_mean']:.1f}" if pd.notna(r.get('gaussian_mean')) else "no fit"
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
