import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Fit the muon lifetime using a linear fit to the logarithm of the histogram
# Theory: N(t) = A * exp(-t/tau) -> ln(N(t)) = ln(A) - t/tau
# This is a linear relationship, where slope = -1/tau
# Process each of the 8 channels separately

# Read histogram data
hist_data = pd.read_csv(r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\Stave Testing\prd_beamline_260326\hists_time_daq0.csv", header=None)

n_channels = hist_data.shape[1]
n_time_bins = hist_data.shape[0]
times = np.arange(n_time_bins)

# Store results for all channels
results = {}

# Process each channel
for channel in range(n_channels):
    counts = hist_data.iloc[:, channel].values
    
    # Filter out zero and negative counts for log fitting
    mask = counts > 0
    times_filtered = times[mask]
    counts_filtered = counts[mask]
    
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
        'log_counts': log_counts
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
        
        # Plot data points
        ax.semilogy(times_filtered, counts_filtered, 'o', alpha=0.7, markersize=4, label='Data')
        
        # Plot fitted curve
        t_fit = np.linspace(times_filtered.min(), times_filtered.max(), 100)
        counts_fit = np.exp(np.polyval(coeffs, t_fit))
        ax.semilogy(t_fit, counts_fit, 'r-', linewidth=2, label=f'Fit: τ={tau:.4f}')
        
        ax.set_xlabel('Time (bin)')
        ax.set_ylabel('Counts')
        ax.set_title(f'Channel {channel}: τ={tau:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'Channel {channel}\n(No data)', ha='center', va='center')
        ax.set_title(f'Channel {channel}')

plt.tight_layout()
plt.show()
