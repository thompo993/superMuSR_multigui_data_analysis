import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Fit the muon lifetime using a linear fit to the logarithm of the histogram
# Theory: N(t) = A * exp(-t/tau) -> ln(N(t)) = ln(A) - t/tau
# This is a linear relationship, where slope = -1/tau
# Process each of the 8 channels separately

# Read histogram data (full spectrum)
hist_data_full = pd.read_csv(r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\Stave Testing\prd_beamline_260326\hists_time_daq0.csv", header=None)
# hist_data_full = pd.read_csv(r"test_parent_dir_1\hists_time_daq0.csv", header=None)

print(f"Full data shape: {hist_data_full.shape}")
print(f"Number of channels: {hist_data_full.shape[1]}")
print(f"Number of time bins: {hist_data_full.shape[0]}")

# Define fitting range (start from bin 1000)
fit_start_bin = 1000

n_channels = hist_data_full.shape[1]
n_time_bins = hist_data_full.shape[0]
times_full = np.arange(n_time_bins)  # Full time array for plotting

# Store results for all channels
results = {}

# Process each channel
for channel in range(n_channels):
    counts_full = hist_data_full.iloc[:, channel].values
    print(f"\nChannel {channel}: Raw counts shape: {counts_full.shape}, Min: {counts_full.min()}, Max: {counts_full.max()}")
    
    # Get data from fit_start_bin onwards
    counts_to_fit = counts_full[fit_start_bin:]
    times_to_fit = times_full[fit_start_bin:]
    
    # Filter out zero and negative counts for log fitting
    mask = counts_to_fit > 0
    times_filtered = times_to_fit[mask]
    counts_filtered = counts_to_fit[mask]
    
    print(f"  Data from bin {fit_start_bin}: {len(times_filtered)} non-zero points")
    
    if len(counts_filtered) < 2:
        print(f"Channel {channel}: Skipped (insufficient data)")
        continue
    
    # Take logarithm of counts
    log_counts = np.log(counts_filtered)
    
    # Fit a linear model: log(N) = a - t/tau, where a is intercept and slope = -1/tau
    coeffs = np.polyfit(times_filtered, log_counts, 1)
    slope, intercept = coeffs
    tau_fit = -1.0 / slope  # Extract lifetime from slope
    A_fit = np.exp(intercept)  # Extract amplitude from intercept
    
    results[channel] = {
        'tau': tau_fit,
        'A': A_fit,
        'slope': slope,
        'intercept': intercept,
        'coeffs': coeffs,
        'times_filtered': times_filtered,
        'counts_filtered': counts_filtered,
        'log_counts': log_counts,
        'counts_full': counts_full
    }
    
    print(f"Channel {channel}: τ={tau_fit:.4f}, A={A_fit:.4f}, Slope={slope:.6f}")

# Calculate average muon lifetime
tau_values = [results[ch]['tau'] for ch in results.keys()]
tau_avg = np.mean(tau_values)
tau_std = np.std(tau_values)
print(f"\nAverage muon lifetime: {tau_avg:.4f} ± {tau_std:.4f} time units")

# Create subplots for all 8 channels
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for channel in range(n_channels):
    ax = axes[channel]
    
    if channel in results:
        res = results[channel]
        times_filtered = res['times_filtered']
        counts_filtered = res['counts_filtered']
        coeffs = res['coeffs']
        tau = res['tau']
        A = res['A']
        counts_full = res['counts_full']
        
        # Plot full spectrum (all data)
        ax.semilogy(times_full, counts_full, 'o', alpha=0.5, markersize=2, label='Full spectrum')
        
        # Plot fitted data points (from bin 1000 onwards)
        ax.semilogy(times_filtered, counts_filtered, 'o', alpha=0.8, markersize=3, color='blue', label='Fitted region')
        
        # Plot fitted curve (from bin 1000 onwards)
        t_fit = np.linspace(times_filtered.min(), times_filtered.max(), 100)
        counts_fit = np.exp(np.polyval(coeffs, t_fit))
        ax.semilogy(t_fit, counts_fit, 'r-', linewidth=2, label=f'Fit: τ={tau:.4f}')
        
        # Mark the fitting start point
        ax.axvline(fit_start_bin, color='g', linestyle='--', alpha=0.5, linewidth=1.5, label=f'Fit start (bin {fit_start_bin})')
        
        ax.set_xlabel('Time (bin)')
        ax.set_ylabel('Counts')
        ax.set_title(f'Channel {channel}: τ={tau:.4f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'Channel {channel}\n(No data)', ha='center', va='center')
        ax.set_title(f'Channel {channel}')

plt.tight_layout()
plt.show()
