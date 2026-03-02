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