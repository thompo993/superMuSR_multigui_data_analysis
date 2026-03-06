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
        print("Digitiser ID outside of expected range (0-4). Please check parent directory.")


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
    Load amplitude histogram CSV — no header, column index = channel.

    Handles two common file formats:
      (a) Clean: 8 columns, no index  → col 0 = ch0 … col 7 = ch7
      (b) With spurious pandas index: 9 columns, col 0 = row index,
          col 1 = ch0 … col 8 = ch7.  Detected by the leading comma /
          NaN in cell [0, 0] and dropped automatically.
    """
    df = pd.read_csv(filepath, header=None, encoding="utf-8")

    # --- Detect & drop spurious row-index column ---
    # Symptom: first cell is NaN (blank before a leading comma)
    if df.shape[1] > 1 and pd.isna(df.iloc[0, 0]):
        candidate = df.iloc[1:, 0]          # skip the NaN header cell
        floated   = pd.to_numeric(candidate, errors="coerce").dropna()
        if len(floated) > 0:
            is_monotone = bool((floated.diff().dropna() >= 0).all())
            is_integers = bool((floated % 1 == 0).all())
            if is_monotone and is_integers:
                df = df.iloc[:, 1:].reset_index(drop=True)

    # --- Drop first row if it is the artefact header (all-NaN or all-zero) ---
    first_row_numeric = pd.to_numeric(df.iloc[0], errors="coerce")
    if first_row_numeric.isna().all() or (first_row_numeric.fillna(0) == 0).all():
        df = df.iloc[1:].reset_index(drop=True)

    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


#=======================================================================
# Gaussian fit
#=======================================================================
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


def fit_gaussian_window(x_win, y_win):
    """
    Fit a Gaussian to a local window.
    Returns (mean, mean_err) or (None, None).
    Rejects fits whose mean falls outside the x window (unphysical result).
    """
    valid = y_win > 0
    xv, yv = x_win[valid], y_win[valid]
    if len(xv) < 4:
        return None, None
    x_lo, x_hi = float(xv[0]), float(xv[-1])
    try:
        p0 = [float(np.max(yv)), float(xv[np.argmax(yv)]),
              max((x_hi - x_lo) / 4.0, 1.0)]
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
    Locate the single most prominent peak (or valley) in y_smooth and fit
    a Gaussian around it.

    Valley robustness
    -----------------
    A valley is only accepted if its smoothed count exceeds 0.5 % of the
    spectrum maximum.  This prevents the noise floor at the far end of the
    spectrum from being mis-labelled as a valley.

    Returns
    -------
    list containing at most one feature dict, or [] if none found.
    """
    n = len(x)
    if n < 5:
        return []

    x_range  = float(x[-1] - x[0])
    half_win = max(x_range * window_frac, 3.0)

    # Invert for valley detection
    signal = -y_smooth if feature_type == "valley" else y_smooth

    sig_range  = float(signal.max() - signal.min())
    prominence = max(sig_range * 0.02, 1e-12)   # ≥ 2 % of dynamic range
    min_dist   = max(2, n // 20)

    indices, _ = find_peaks(signal, prominence=prominence, distance=min_dist)

    # Drop features at or below min_x
    if len(indices):
        indices = indices[x[indices] > min_x]

    # For valleys: reject candidates on the noise floor (< 0.5 % of max)
    if feature_type == "valley" and len(indices):
        floor_threshold = float(y_smooth.max()) * 0.005
        indices = indices[y_smooth[indices] > floor_threshold]

    if len(indices) == 0:
        return []

    # Single most prominent feature
    best_idx = int(indices[np.argmax(signal[indices])])
    feat_x   = float(x[best_idx])

    # Local fitting window
    mask  = (x >= feat_x - half_win) & (x <= feat_x + half_win)
    x_win = x[mask]
    y_win = y_norm[mask]

    mean, mean_err = fit_gaussian_window(x_win, y_win)

    return [{
        "feature_type":      feature_type,
        "feature_rank":      1,
        "bin_index":         int(best_idx),
        "smoothed_x":        feat_x,
        "smoothed_y":        float(y_smooth[best_idx]),
        "gaussian_mean":     mean,
        "gaussian_mean_err": mean_err,
        "n_fit_points":      int(mask.sum()),
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
      4. Detect single peak and single valley; fit Gaussians
      5. Write per-channel, per-digitiser, and master CSV files

    Output files (written to *output_dir*)
    ───────────────────────────────────────
    daq{N}_ch{C}_peaks.csv        peak fit for digitiser N, channel C
    daq{N}_ch{C}_valleys.csv      valley fit for digitiser N, channel C
    daq{N}_all_channels.csv       combined peak+valley for digitiser N
    all_digitisers_summary.csv    one row per feature, every DAQ & channel
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

            # ── Savitzky-Golay smooth ─────────────────────────────────
            win = min(savgol_window, len(normed_a) - 1)
            if win % 2 == 0:
                win -= 1
            win = max(win, savgol_poly + 2)
            smooth_a = savgol_filter(normed_a, window_length=win,
                                     polyorder=savgol_poly)

            # ── detect & fit ──────────────────────────────────────────
            peaks   = _fit_features(x_a, smooth_a, normed_a, "peak",
                                    window_frac=0.10, min_x=min_x)
            valleys = _fit_features(x_a, smooth_a, normed_a, "valley",
                                    window_frac=0.10, min_x=min_x)

            # Remap indices from the trimmed array back to full coordinates
            for feat_list in (peaks, valleys):
                for feat in feat_list:
                    feat["bin_index"]  = int(feat["bin_index"]  + start)
                    feat["smoothed_x"] = float(feat["smoothed_x"] + start)
                    if feat["gaussian_mean"] is not None:
                        feat["gaussian_mean"] = float(feat["gaussian_mean"] + start)

            # ── assemble rows ─────────────────────────────────────────
            base    = {"digitiser_id": dig_id, "channel": ch,
                       "normalised": norm_applied, "norm_factor": norm_factor}
            ch_rows = [{**base, **f} for f in peaks + valleys]
            dig_rows.extend(ch_rows)
            all_rows.extend(ch_rows)

            # ── per-channel CSVs ──────────────────────────────────────
            pk_df = pd.DataFrame([r for r in ch_rows if r["feature_type"] == "peak"])
            vl_df = pd.DataFrame([r for r in ch_rows if r["feature_type"] == "valley"])

            pk_path = output_dir / f"daq{dig_id}_ch{ch}_peaks.csv"
            vl_path = output_dir / f"daq{dig_id}_ch{ch}_valleys.csv"
            pk_df.to_csv(pk_path, index=False)
            vl_df.to_csv(vl_path, index=False)

            print(f"  ch{ch}: {len(pk_df)} peak(s), {len(vl_df)} valley/ies "
                  f"→ {pk_path.name}, {vl_path.name}")

        # ── per-digitiser combined CSV ────────────────────────────────
        if dig_rows:
            dig_path = output_dir / f"daq{dig_id}_all_channels.csv"
            pd.DataFrame(dig_rows).to_csv(dig_path, index=False)
            print(f"[DAQ {dig_id}] Combined CSV → {dig_path.name}")

    # ── master summary CSV ────────────────────────────────────────────
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
