import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

# importing and reading data, and selecting ch0
df = pd.read_csv(
    r"test_parent_dir\hists_amplitude_daq0.csv",header=None)
ch0  = df.iloc[:,0]

print(ch0)

# Convert to numpy array for easier manipulation
data = ch0.values
x = np.arange(len(data))

# Select data range 300 < x < 4000
mask = (x > 300) & (x < 4000)
x_range = x[mask]
data_range = data[mask]

# Smooth the data using Savitzky-Golay filter
window_length = min(51, len(data_range))

# window_length must be odd
if window_length % 2 == 0:
    window_length -= 1

# Smooth the data using Savitzky-Golay filter
n = len(data_range)

if n < 5:
    raise ValueError(f"Not enough data points for smoothing: {n}")

# window_length must be odd and <= data length
window_length = min(51, n)

if window_length % 2 == 0:
    window_length -= 1

# polyorder must be < window_length
polyorder = min(3, window_length - 2)

data_smoothed = savgol_filter(data_range, window_length=window_length, polyorder=polyorder)

# Find peaks in the smoothed data
peaks, properties = find_peaks(data_smoothed, height=np.max(data_smoothed)*0.1, distance=10)

# Filter peaks to only include those after x=500
peaks_filtered = peaks[x_range[peaks] > 500]

# Select only the most prominent peak
if len(peaks_filtered) > 0:
    prominent_peak_idx = peaks_filtered[np.argmax(properties['peak_heights'][np.isin(peaks, peaks_filtered)])]
    
    # Define a Gaussian function for fitting
    def gaussian(x, amplitude, mean, sigma):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
    
    # Extract data around the peak (±100 bins)
    peak_window = 100
    peak_x_value = x_range[prominent_peak_idx]
    mask_around_peak = (x_range >= peak_x_value - peak_window) & (x_range <= peak_x_value + peak_window)
    x_peak_region = x_range[mask_around_peak]
    data_peak_region = data_smoothed[mask_around_peak]
    
    # Better initial parameter guesses
    peak_height = data_peak_region[np.argmax(data_peak_region)]
    peak_center = x_peak_region[np.argmax(data_peak_region)]
    # Estimate sigma from FWHM (Full Width at Half Maximum)
    half_max = peak_height / 2
    above_half = data_peak_region > half_max
    if np.sum(above_half) > 1:
        fwhm = x_peak_region[above_half][-1] - x_peak_region[above_half][0]
        sigma_guess = fwhm / 2.355
    else:
        sigma_guess = 10
    
    p0 = [peak_height, peak_center, sigma_guess]
    
    # Fit the Gaussian to data around the peak
    try:
        popt, pcov = curve_fit(gaussian, x_peak_region, data_peak_region, p0=p0, maxfev=10000)
        amplitude, mean, sigma = popt
        
        # Generate fitted curve for the entire range and extract the peak value
        fitted_data = gaussian(x_range, amplitude, mean, sigma)
        fitted_data_peak = (amplitude, mean)









        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(x_range, data_range, 'b-', label='Original Data', linewidth=1, alpha=0.7)
        plt.plot(x_range, data_smoothed, 'c-', label='Smoothed Data', linewidth=2)
        plt.plot(x_range, fitted_data, 'r--', label='Gaussian Fit', linewidth=2)
        plt.plot(x_range[prominent_peak_idx], data_smoothed[prominent_peak_idx], 'go', label='Selected Inital Peak', markersize=10)
        plt.plot(fitted_data_peak[1], fitted_data_peak[0], 'x', label='Guassian Fitted Peak', markersize=10, markeredgewidth=2, color='magenta')
        plt.axvline(peak_x_value - peak_window, color='gray', linestyle=':', alpha=0.5, label='Fit Region')
        plt.axvline(peak_x_value + peak_window, color='gray', linestyle=':', alpha=0.5)
        plt.xlabel('Bin')
        plt.ylabel('Amplitude')
        plt.title('Peak Finding and Gaussian Fitting (300 < x < 4000, peak after x=500)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Print fitted parameters
        print(f"Fitted Parameters:")
        print(f"Amplitude: {amplitude:.2f}")
        print(f"Mean: {mean:.2f}")
        print(f"Sigma (std dev): {sigma:.2f}")
    except RuntimeError as e:
        print(f"Fit failed: {e}")
else:
    print("No peaks detected after x=500")
