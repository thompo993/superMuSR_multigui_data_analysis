#=======================================================================
# imports
#=======================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
import pandas as pd
from pathlib import Path
import os
import glob
import warnings
import re
from datetime import datetime
from pprint import pprint
import json

warnings.filterwarnings("ignore")



#=======================================================================
# inputs
#=======================================================================
parent_dir = r"./test_parent_dir"



#=======================================================================
# workflow
#=======================================================================
def extract_daq_from_filename(filename):
    """
    Extract last item of string as this is the digitser ID.
    """
    stem = Path(str(filename)).stem
    last_char = stem[-1]

    if last_char.isdigit() and 0 <= int(last_char) <= 4:
        return int(last_char)
    else: print("Digitser ID outside of Expected Range (0-4. Please Check Parent Directory)")



def group_files_by_digitiser(parent_directory):
    """
    Groups files by digitiser ID.

    Returns:
        groups: dict {digitiser_id: [filepaths]}
        file_digitiser_map: dict {filepath: digitiser_id}
    """
    parent_directory = Path(parent_directory)

    groups = {}
    file_digitiser_map = {}

    for file in parent_directory.iterdir():
        if not file.is_file():
            continue

        digitiser_id = extract_daq_from_filename(file.name)

        if digitiser_id is None:
            print(f"No digitser Id found for: {file.name} ")
            # skip files that do not match expected DAQ format
            continue

        # file to digitiser mapping
        file_digitiser_map[file] = digitiser_id

        # Add to group
        if digitiser_id not in groups:
            groups[digitiser_id] = []

        groups[digitiser_id].append(file)

    return groups, file_digitiser_map


# printing digitier group assignemnt and counts
groups, file_digitiser_map = group_files_by_digitiser(parent_dir)

pprint({k: [p.name for p in v] for k, v in groups.items()})

print("\nCounts per digitiser:")
for digitiser_id, files in groups.items():
    print(f"Digitiser {digitiser_id}: {len(files)} files")

print("\nFile Digitiser mapping:")
for file, digitiser_id in file_digitiser_map.items():
    print(f"{file.name} Digitiser {digitiser_id}")

def extract_and_read_hk_from_groups(groups):
    """
    Extract HK file and trigger_counter for each digitiser.
    """

    hk_map = {}
    trigger_counts = {}

    for digitiser_id, files in groups.items():
        print(f"\n--- Checking Digitiser {digitiser_id} ---")

        hk_file = None

        # ---------- Find HK file ----------
        for f in files:
            name = f.name.lower()

            if "hk" in name or f.suffix.lower() == ".set" or f.suffix.lower() == ".json":
                hk_file = f
                print(f"Found HK file: {hk_file.name}")
                break

        hk_map[digitiser_id] = hk_file

        if hk_file is None:
            print("No HK file found.")
            trigger_counts[digitiser_id] = None
            continue

        trigger_value = None

        # ---------- JSON HK ----------
        if hk_file.suffix.lower() == ".json":
            try:
                with open(hk_file, "r") as fp:
                    data = json.load(fp)

                print("JSON contents:", data)

                # Handle nested dictionaries recursively
                def find_trigger(d):
                    for k, v in d.items():
                        if "trigger" in k.lower():
                            return v
                        if isinstance(v, dict):
                            result = find_trigger(v)
                            if result is not None:
                                return result
                    return None

                trigger_value = find_trigger(data)

                print(f"Extracted trigger_counter: {trigger_value}")

            except Exception as e:
                print("JSON read error:", e)
                trigger_value = None

        # ---------- .set HK ----------
        elif hk_file.suffix.lower() == ".set":
            try:
                with open(hk_file, "r") as fp:
                    lines = fp.readlines()

                for line in lines:
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if "trigger" in k.lower():
                            trigger_value = v.strip()
                            break

                print(f"Extracted trigger_counter: {trigger_value}")

            except Exception as e:
                print("SET read error:", e)
                trigger_value = None

        else:
            print("Unsupported HK file type.")
            trigger_value = None

        trigger_counts[digitiser_id] = trigger_value

    return hk_map, trigger_counts

def build_digitiser_summary(groups):
    """
    Builds a structured list of dictionaries for each digitiser containing:
        - HK file
        - trigger_counter (normalisation)
        - time histogram CSV
        - amplitude histogram CSV

    Returns:
        summary_list: list of dictionaries (one per digitiser)
    """

    hk_map, trigger_counts = extract_and_read_hk_from_groups(groups)

    summary_list = []

    for digitiser_id, files in groups.items():

        time_hist_file = None
        amp_hist_file = None

        # --- Identify histogram files ---
        for f in files:
            name = f.name.lower()

            if f.suffix.lower() == ".csv":

                if "time" in name:
                    time_hist_file = f

                elif "amp" in name or "amplitude" in name:
                    amp_hist_file = f

        digitiser_dict = {
            "digitiser_id": digitiser_id,
            "hk_file": hk_map.get(digitiser_id),
            "trigger_counter": trigger_counts.get(digitiser_id),
            "time_histogram": time_hist_file,
            "amplitude_histogram": amp_hist_file,
        }

        summary_list.append(digitiser_dict)

    return summary_list


digitiser_summary = build_digitiser_summary(groups)

print("\nDigitiser Summary Structure:\n")

for d in digitiser_summary:
    print(f"Digitiser {d['digitiser_id']}")
    print(f"  HK File: {d['hk_file']}")
    print(f"  Trigger Counter: {d['trigger_counter']}")
    print(f"  Time Histogram: {d['time_histogram']}")
    print(f"  Amplitude Histogram: {d['amplitude_histogram']}")
    print("="* 50)

    #=======================================================================
# Amplitude PHS Plotting — 4 Digitisers × 8 Channels (32-panel figure)
#=======================================================================

N_CHANNELS = 8
COLOURS     = plt.cm.tab10.colors   # one distinct colour per channel


def load_amplitude_csv(filepath):
    """
    Load an amplitude histogram CSV (UTF-8, no header).
    Column index == channel number  (col 0 → ch0, col 7 → ch7).

    Returns:
        df : pd.DataFrame  shape (n_bins, n_channels)
    """
    df = pd.read_csv(filepath, header=None, encoding="utf-8")
    return df


def plot_amplitude_phs(digitiser_summary, n_channels=N_CHANNELS, normalise=True):
    """
    Plot a (n_digitisers × n_channels) grid of PHS panels.

    Parameters
    ----------
    digitiser_summary : list of dicts  — from build_digitiser_summary()
    n_channels        : int            — channels per digitiser (default 8)
    normalise         : bool           — divide counts by trigger_counter
    """

    # Consistent row ordering
    sorted_summary = sorted(digitiser_summary, key=lambda d: d["digitiser_id"])
    n_digs = len(sorted_summary)

    fig, axes = plt.subplots(
        nrows=n_digs,
        ncols=n_channels,
        figsize=(n_channels * 3.2, n_digs * 2.8),
        sharex=False,
        sharey=False,
    )

    # Guarantee axes is always 2-D
    if n_digs == 1:
        axes = np.array([axes])
    if n_channels == 1:
        axes = axes[:, np.newaxis]

    for row_idx, entry in enumerate(sorted_summary):

        dig_id          = entry["digitiser_id"]
        amp_file        = entry["amplitude_histogram"]
        trigger_counter = entry["trigger_counter"]

        # ------------------------------------------------------------------
        # Resolve normalisation factor
        # ------------------------------------------------------------------
        norm_factor = 1.0
        if normalise and trigger_counter is not None:
            try:
                norm_factor = float(trigger_counter)
                if norm_factor == 0:
                    print(f"  [DAQ {dig_id}] trigger_counter is 0 — skipping normalisation.")
                    norm_factor = 1.0
            except (ValueError, TypeError):
                print(f"  [DAQ {dig_id}] Cannot convert trigger_counter "
                      f"'{trigger_counter}' to float — skipping normalisation.")

        # ------------------------------------------------------------------
        # Load CSV
        # ------------------------------------------------------------------
        df = None
        if amp_file is not None:
            try:
                df = load_amplitude_csv(amp_file)
                print(f"  [DAQ {dig_id}] Loaded {amp_file.name}  "
                      f"({df.shape[0]} bins × {df.shape[1]} channels)")
            except Exception as exc:
                print(f"  [DAQ {dig_id}] Failed to read amplitude CSV: {exc}")

        # ------------------------------------------------------------------
        # Draw each channel subplot
        # ------------------------------------------------------------------
        for ch in range(n_channels):
            ax     = axes[row_idx, ch]
            colour = COLOURS[ch % len(COLOURS)]

            if df is not None and ch < df.shape[1]:
                counts    = df.iloc[:, ch].values.astype(float) / norm_factor
                bin_edges = np.arange(len(counts))

                ax.step(bin_edges, counts, where="mid",
                        color=colour, linewidth=0.9)
                ax.fill_between(bin_edges, counts, step="mid",
                                color=colour, alpha=0.20)
                ax.yaxis.set_tick_params(labelsize=6)
                ax.xaxis.set_tick_params(labelsize=6)

            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="grey")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylimits(0, 1)  # default y-limits for empty plots

            # Column headers — top row only
            if row_idx == 0:
                ax.set_title(f"Ch {ch}", fontsize=9, fontweight="bold", pad=4)

            # Row labels — left column only
            if ch == 0:
                norm_label = f"\n(÷ {int(norm_factor):,})" if norm_factor != 1.0 else ""
                ax.set_ylabel(f"DAQ {dig_id}{norm_label}", fontsize=8)

            # x-axis label — bottom row only
            if row_idx == n_digs - 1:
                ax.set_xlabel("Amplitude bin", fontsize=7)

    fig.suptitle(
        "Amplitude Pulse-Height Spectra — All DAQs & Channels",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig("amplitude_phs_all.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved → amplitude_phs_all.png")
    plt.show()


# ---- Run ----
plot_amplitude_phs(digitiser_summary, n_channels=N_CHANNELS, normalise=True)