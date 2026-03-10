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
parent_dir = r"./test_parent_dir_1"
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
    seen_ids = set()
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
    """
    df = pd.read_csv(filepath, header=None, encoding="utf-8")

    if df.shape[1] > 1 and pd.isna(df.iloc[0, 0]):
        candidate = df.iloc[1:, 0]
        floated   = pd.to_numeric(candidate, errors="coerce").dropna()
        if len(floated) > 0:
            is_monotone = bool((floated.diff().dropna() >= 0).all())
            is_integers = bool((floated % 1 == 0).all())
            if is_monotone and is_integers:
                df = df.iloc[:, 1:].reset_index(drop=True)

    first_row = pd.to_numeric(df.iloc[0], errors="coerce")
    if first_row.isna().all() or (first_row.fillna(0) == 0).all():
        df = df.iloc[1:].reset_index(drop=True)

    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


#=======================================================================
# Gaussian model
#=======================================================================
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


#=======================================================================
# Peak detection & Gaussian fit  (ported from user's single-channel script)
#=======================================================================
def detect_peak(x_range, data_smoothed, data_raw, min_peak_x=500):
    """
    Find the single most prominent peak above *min_peak_x* and fit a Gaussian.

    Parameters
    ----------
    x_range       : 1-D array of bin indices (already sliced to search window)
    data_smoothed : smoothed counts corresponding to x_range
    data_raw      : raw (or normalised) counts corresponding to x_range
    min_peak_x    : bins below this value are excluded from peak candidates

    Returns
    -------
    dict with keys matching the rest of the pipeline, or None if no peak found.
    """
    peaks, _ = find_peaks(
        data_smoothed,
        height=np.max(data_smoothed) * 0.1,
        distance=10,
    )

    # Keep only peaks beyond min_peak_x
    peaks_filtered = peaks[x_range[peaks] > min_peak_x]

    if len(peaks_filtered) == 0:
        return None

    # Most prominent (highest smoothed value)
    prominent_idx = peaks_filtered[np.argmax(data_smoothed[peaks_filtered])]
    peak_x_value  = x_range[prominent_idx]

    # --- Gaussian fit in a ±100-bin window around the detected peak ---
    peak_window = 100
    mask_peak   = (x_range >= peak_x_value - peak_window) & \
                  (x_range <= peak_x_value + peak_window)

    x_fit   = x_range[mask_peak]
    y_fit   = data_smoothed[mask_peak]        # fit on smoothed data (same as user script)

    peak_height  = float(np.max(y_fit))
    peak_center  = float(x_fit[np.argmax(y_fit)])

    # Estimate sigma from FWHM
    half_max   = peak_height / 2.0
    above_half = y_fit > half_max
    if np.sum(above_half) > 1:
        fwhm        = float(x_fit[above_half][-1] - x_fit[above_half][0])
        sigma_guess = fwhm / 2.355
    else:
        sigma_guess = 10.0

    p0 = [peak_height, peak_center, sigma_guess]

    mean, mean_err = None, None
    try:
        popt, pcov   = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=10000)
        mean         = float(popt[1])
        var          = float(pcov[1, 1])
        mean_err     = float(np.sqrt(var)) if var >= 0 else None
    except (RuntimeError, ValueError):
        pass

    return {
        "feature_type":      "peak",
        "feature_rank":      1,
        "bin_index":         int(prominent_idx),          # local index inside x_range
        "smoothed_x":        float(peak_x_value),
        "smoothed_y":        float(data_smoothed[prominent_idx]),
        "gaussian_mean":     mean,
        "gaussian_mean_err": mean_err,
        "n_fit_points":      int(np.sum(mask_peak)),
        "fallback":          False,
    }


#=======================================================================
# Valley detection  (ported from user's single-channel script)
#=======================================================================
def detect_valley(x_range, data_smoothed, peak_x_value,
                  valley_search_width=400):
    """
    Find the valley as the minimum of the smoothed spectrum in the window
    [peak_x_value - valley_search_width, peak_x_value].

    Returns a dict or None.
    """
    valley_mask = (x_range < peak_x_value) & \
                  (x_range > peak_x_value - valley_search_width)

    if valley_mask.sum() == 0:
        return None

    x_v   = x_range[valley_mask]
    y_v   = data_smoothed[valley_mask]
    v_idx = int(np.argmin(y_v))

    valley_x = float(x_v[v_idx])
    valley_y = float(y_v[v_idx])

    # Local index within x_range
    local_idx = int(np.where(x_range == x_v[v_idx])[0][0])

    return {
        "feature_type":      "valley",
        "feature_rank":      1,
        "bin_index":         local_idx,
        "smoothed_x":        valley_x,
        "smoothed_y":        valley_y,
        "gaussian_mean":     None,      # no Gaussian fit on valley in user script
        "gaussian_mean_err": None,
        "n_fit_points":      0,
        "fallback":          False,
    }


#=======================================================================
# Main analysis
#=======================================================================
def analyse_amplitude_phs(digitiser_summary,
                           n_channels    = N_CHANNELS,
                           normalise     = NORMALISE,
                           output_dir    = OUTPUT_DIR,
                           savgol_window = 51,
                           savgol_poly   = 3,
                           search_x_min  = 300,
                           search_x_max  = 4000,
                           min_peak_x    = 500,
                           valley_search_width = 400):
    """
    For every digitiser × channel:
      1. Load amplitude histogram CSV
      2. Optionally normalise by trigger_counter
      3. Slice to [search_x_min, search_x_max]; smooth with Savitzky-Golay
      4. Detect single peak (find_peaks → Gaussian fit) and valley (argmin)
      5. Write per-channel, per-digitiser, and master CSV files

    Output files (written to *output_dir*)
    ───────────────────────────────────────
    daq{N}_ch{C}_peaks.csv
    daq{N}_ch{C}_valleys.csv
    daq{N}_all_channels.csv
    all_digitisers_summary.csv
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

            # ── Slice to search window (matches user script) ─────────
            mask_search = (x_full > search_x_min) & (x_full < search_x_max)
            x_range     = x_full[mask_search]
            data_range  = normed[mask_search]

            if len(data_range) < 5:
                print(f"  [DAQ {dig_id}] ch{ch}: too few points in search "
                      "window — skipping.")
                continue

            # ── Savitzky-Golay smooth ────────────────────────────────
            n   = len(data_range)
            win = min(savgol_window, n)
            if win % 2 == 0:
                win -= 1
            poly    = min(savgol_poly, win - 2)
            win     = max(win, poly + 2)
            smooth  = savgol_filter(data_range, window_length=win,
                                    polyorder=poly)

            # ── Peak detection ───────────────────────────────────────
            peak_result = detect_peak(x_range, smooth, data_range,
                                      min_peak_x=min_peak_x)

            # ── Valley detection (only if a peak was found) ──────────
            valley_result = None
            if peak_result is not None:
                valley_result = detect_valley(
                    x_range, smooth,
                    peak_result["smoothed_x"],
                    valley_search_width=valley_search_width,
                )

            # ── Assemble rows ────────────────────────────────────────
            base = {
                "digitiser_id": dig_id,
                "channel":      ch,
                "normalised":   norm_applied,
                "norm_factor":  norm_factor,
            }

            ch_rows = []
            for feat in ([peak_result] if peak_result else []) + \
                        ([valley_result] if valley_result else []):
                ch_rows.append({**base, **feat})

            dig_rows.extend(ch_rows)
            all_rows.extend(ch_rows)

            # ── Per-channel CSVs ─────────────────────────────────────
            pk_df = pd.DataFrame([r for r in ch_rows
                                  if r.get("feature_type") == "peak"])
            vl_df = pd.DataFrame([r for r in ch_rows
                                  if r.get("feature_type") == "valley"])

            pk_path = output_dir / f"daq{dig_id}_ch{ch}_peaks.csv"
            vl_path = output_dir / f"daq{dig_id}_ch{ch}_valleys.csv"
            pk_df.to_csv(pk_path, index=False)
            vl_df.to_csv(vl_path, index=False)

            # ── Diagnostic print ─────────────────────────────────────
            pk_info = "NO PEAK"
            if peak_result:
                gm = (f"{peak_result['gaussian_mean']:.1f}"
                      if peak_result["gaussian_mean"] is not None else "no fit")
                pk_info = (f"peak@{peak_result['smoothed_x']:.0f}"
                           f"(gaus={gm})")

            vl_info = "no valley"
            if valley_result:
                vl_info = f"valley@{valley_result['smoothed_x']:.0f}"

            print(f"  ch{ch}: {pk_info}, {vl_info}")

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
    n_channels          = N_CHANNELS,
    normalise           = NORMALISE,
    output_dir          = OUTPUT_DIR,
    savgol_window       = 51,
    savgol_poly         = 3,
    search_x_min        = 300,
    search_x_max        = 4000,
    min_peak_x          = 500,
    valley_search_width = 400,
)