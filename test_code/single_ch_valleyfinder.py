import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

# importing and reading data, and selecting ch0
df = pd.read_csv(r"test_parent_dir\hists_amplitude_daq0.csv", header=None)
ch0 = df.iloc[:,6]

# Convert to numpy array
data = ch0.values
x = np.arange(len(data))

# Select data range
mask = (x > 300) & (x < 4000)
x_range = x[mask]
data_range = data[mask]

# --- Smooth data ---
n = len(data_range)

if n < 5:
    raise ValueError("Not enough data points")

window_length = min(51, n)

if window_length % 2 == 0:
    window_length -= 1

polyorder = min(3, window_length - 2)

data_smoothed = savgol_filter(data_range, window_length=window_length, polyorder=polyorder)

# --- Find peaks ---
peaks, properties = find_peaks(data_smoothed,
                               height=np.max(data_smoothed)*0.1,
                               distance=10)

# Keep peaks after x=500
peaks_filtered = peaks[x_range[peaks] > 500]

if len(peaks_filtered) > 0:

    prominent_peak_idx = peaks_filtered[
        np.argmax(data_smoothed[peaks_filtered])
    ]

    peak_x_value = x_range[prominent_peak_idx]

    # --- Gaussian model ---
    def gaussian(x, amplitude, mean, sigma):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    # --- Peak fitting region ---
    peak_window = 100

    mask_peak = (x_range >= peak_x_value - peak_window) & (x_range <= peak_x_value + peak_window)

    x_peak_region = x_range[mask_peak]
    data_peak_region = data_smoothed[mask_peak]

    peak_height = np.max(data_peak_region)
    peak_center = x_peak_region[np.argmax(data_peak_region)]

    # Estimate sigma from FWHM
    half_max = peak_height / 2
    above_half = data_peak_region > half_max

    if np.sum(above_half) > 1:
        fwhm = x_peak_region[above_half][-1] - x_peak_region[above_half][0]
        sigma_guess = fwhm / 2.355
    else:
        sigma_guess = 10

    p0 = [peak_height, peak_center, sigma_guess]

    try:

        popt, pcov = curve_fit(gaussian, x_peak_region, data_peak_region,
                               p0=p0, maxfev=10000)

        amplitude, mean, sigma = popt

        fitted_data = gaussian(x_range, amplitude, mean, sigma)

        # -----------------------------
        # Valley detection (clean)
        # -----------------------------

        valley_search = (x_range < peak_x_value) & (x_range > peak_x_value - 400)

        x_valley_search = x_range[valley_search]
        y_valley_search = data_smoothed[valley_search]

        valley_idx = np.argmin(y_valley_search)

        valley_x = x_valley_search[valley_idx]
        valley_y = y_valley_search[valley_idx]

        # -----------------------------
        # Plotting
        # -----------------------------

        plt.figure(figsize=(12,6))

        plt.plot(x_range, data_range,
                 label="Original Data", linewidth=1, alpha=0.6)

        plt.plot(x_range, data_smoothed,
                 label="Smoothed Data", linewidth=2)

        plt.plot(x_range, fitted_data,
                 '--', label="Gaussian Fit", linewidth=2)

        # Initial peak
        plt.plot(x_range[prominent_peak_idx],
                 data_smoothed[prominent_peak_idx],
                 'go', label="Detected Peak", markersize=10)

        # Fitted peak
        plt.plot(mean, amplitude,
                 'mx', label="Gaussian Peak", markersize=12, markeredgewidth=2)

        # Valley
        plt.plot(valley_x, valley_y,
                 'ro', label="Detected Valley", markersize=10)

        # Fit region markers
        plt.axvline(peak_x_value - peak_window, linestyle=":", color="gray")
        plt.axvline(peak_x_value + peak_window, linestyle=":", color="gray")

        plt.xlabel("Bin")
        plt.ylabel("Amplitude")

        plt.title("Peak Detection + Gaussian Fit + Valley Detection")

        plt.legend()
        plt.grid(True)
        plt.show()

        # Output parameters
        print("Fitted Parameters:")
        print(f"Amplitude: {amplitude:.2f}")
        print(f"Mean: {mean:.2f}")
        print(f"Sigma: {sigma:.2f}")

        print(f"Valley Location: x = {valley_x:.2f}, y = {valley_y:.2f}")

    except RuntimeError as e:
        print("Fit failed:", e)

else:
    print("No peaks detected after x=500")