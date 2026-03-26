import pandas as pd
import numpy as np

# Read histogram data
hist_data = pd.read_csv(r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\Stave Testing\prd_beamline_260326\hists_time_daq0.csv", header=None)

print(f"Shape: {hist_data.shape}")
print(f"\nFirst 100 rows of channel 0:\n{hist_data.iloc[:100, 0].values}")

# Find where data actually starts (first non-zero value)
channel_0 = hist_data.iloc[:, 0].values
first_nonzero = np.where(channel_0 > 0)[0]
if len(first_nonzero) > 0:
    print(f"\nFirst non-zero at bin: {first_nonzero[0]}")
    print(f"Last non-zero at bin: {first_nonzero[-1]}")
    print(f"Range: {first_nonzero[-1] - first_nonzero[0]} bins")
    
    # Show data around the first and last non-zero values
    start_idx = max(0, first_nonzero[0] - 5)
    print(f"\nAround first non-zero (bins {start_idx}-{first_nonzero[0]+10}):")
    print(hist_data.iloc[start_idx:first_nonzero[0]+10, 0].values)
    
    end_idx = first_nonzero[-1]
    print(f"\nAround last non-zero (bins {end_idx-10}-{end_idx}):")
    print(hist_data.iloc[end_idx-10:end_idx+1, 0].values)
