#=======================================================================
# plot_phs_results.py
# Reads CSVs produced by amplitude_phs_analysis.py and plots all
# 32 channels (4 DAQs x 8 channels) in a single figure, overlaying
# detected peak / valley positions with Gaussian-fit error bands.
#=======================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

#=======================================================================
# inputs  -- edit these to match your setup
#=======================================================================
RESULTS_DIR   = r"./data"      # folder written by analysis script
RAW_DATA_DIR  = r"./test_parent_dir" # original amplitude CSVs (for spectrum)
OUTPUT_FIGURE = r"./phs_all_channels.png"

N_CHANNELS = 8
NORMALISE  = True    # must match what was used in the analysis script

LOG_SCALE    = True
SHOW_PEAKS   = True
SHOW_VALLEYS = True

# One colour per channel, consistent across all DAQ rows
CH_COLOURS = [
    "#ffff00", "#ff0000", "#00ff00", "#0000ff",
    "#03fdfd", "#ff00ff", "#ffffff", "#ffa500",
]
PEAK_COLOUR   = "#E63946"   # red  — peaks
VALLEY_COLOUR = "#457B9D"   # blue — valleys


#=======================================================================
# helpers
#=======================================================================
def load_spectrum(raw_data_dir, dig_id):
    """Find and load the amplitude CSV for a given digitiser."""
    raw_data_dir = Path(raw_data_dir)
    candidates = (list(raw_data_dir.glob(f"*amp*{dig_id}*.csv")) +
                  list(raw_data_dir.glob(f"*amplitude*{dig_id}*.csv")))
    if not candidates:
        candidates = [
            f for f in raw_data_dir.glob("*.csv")
            if f.stem[-1] == str(dig_id)
            and ("amp" in f.name.lower() or "amplitude" in f.name.lower())
        ]
    if not candidates:
        return None
    try:
        return pd.read_csv(candidates[0], header=None, encoding="utf-8")
    except Exception as exc:
        print(f"  Could not load spectrum for DAQ {dig_id}: {exc}")
        return None


def load_fit_results(results_dir, dig_id, ch, feature_type):
    """Load daq{N}_ch{C}_{peaks|valleys}.csv — returns empty DataFrame if missing."""
    path = Path(results_dir) / f"daq{dig_id}_ch{ch}_{feature_type}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_norm_factor(results_dir, dig_id):
    """Read norm_factor from the first available peaks CSV for this DAQ."""
    for ch in range(N_CHANNELS):
        pk = load_fit_results(results_dir, dig_id, ch, "peaks")
        if not pk.empty and "norm_factor" in pk.columns:
            return float(pk["norm_factor"].iloc[0])
    return 1.0


def discover_daq_ids(results_dir):
    """Return sorted list of DAQ IDs present in the results folder."""
    ids = set()
    for p in Path(results_dir).glob("daq*_ch*_peaks.csv"):
        try:
            ids.add(int(p.name.split("_")[0].replace("daq", "")))
        except ValueError:
            pass
    return sorted(ids)


def _nan(val):
    """True if val is None or float NaN."""
    try:
        return val is None or (isinstance(val, float) and np.isnan(val))
    except Exception:
        return True


#=======================================================================
# main plotting function
#=======================================================================
def plot_all_channels(
        results_dir   = RESULTS_DIR,
        raw_data_dir  = RAW_DATA_DIR,
        output_figure = OUTPUT_FIGURE,
        n_channels    = N_CHANNELS,
        log_scale     = LOG_SCALE,
        show_peaks    = SHOW_PEAKS,
        show_valleys  = SHOW_VALLEYS,
        normalise     = NORMALISE,
):
    results_dir   = Path(results_dir)
    raw_data_dir  = Path(raw_data_dir)
    output_figure = Path(output_figure)

    daq_ids = discover_daq_ids(results_dir)
    if not daq_ids:
        print(f"No result CSVs found in {results_dir}. "
              "Run amplitude_phs_analysis.py first.")
        return

    n_digs = len(daq_ids)
    print(f"Found DAQ IDs: {daq_ids}")

    # ── figure / axes ────────────────────────────────────────────────
    fig, axes = plt.subplots(
        nrows=n_digs, ncols=n_channels,
        figsize=(n_channels * 3.0, n_digs * 2.6),
        sharex=False, sharey=False,
    )
    fig.patch.set_facecolor("#0F1117")

    if n_digs == 1:
        axes = np.array([axes])
    if n_channels == 1:
        axes = axes[:, np.newaxis]

    for row_idx, dig_id in enumerate(daq_ids):

        spec_df     = load_spectrum(raw_data_dir, dig_id)
        norm_factor = get_norm_factor(results_dir, dig_id)

        for ch in range(n_channels):
            ax     = axes[row_idx, ch]
            colour = CH_COLOURS[ch % len(CH_COLOURS)]

            # ── panel styling ────────────────────────────────────────
            ax.set_facecolor("#161B22")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363D")
                spine.set_linewidth(0.7)
            ax.tick_params(colors="#8B949E", labelsize=5.5, length=3, width=0.6)

            # ── spectrum ─────────────────────────────────────────────
            plotted = False
            if spec_df is not None and ch < spec_df.shape[1]:
                raw     = spec_df.iloc[:, ch].values.astype(float)
                counts  = raw / norm_factor if (normalise and norm_factor != 1.0) else raw
                bins    = np.arange(len(counts), dtype=float)
                display = np.where(counts > 0, counts, np.nan)

                ax.step(bins, display, where="mid",
                        color=colour, linewidth=0.85, alpha=0.9)
                ax.fill_between(bins, display, step="mid",
                                color=colour, alpha=0.12)

                if log_scale:
                    ax.set_yscale("log")
                    ax.yaxis.set_major_locator(
                        ticker.LogLocator(base=10, numticks=4))
                    ax.yaxis.set_minor_locator(ticker.NullLocator())
                plotted = True

            if not plotted:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7,
                        color="#484F58", fontstyle="italic")

            # ── peak overlays ────────────────────────────────────────
            if show_peaks:
                pk_df = load_fit_results(results_dir, dig_id, ch, "peaks")
                for _, row in pk_df.iterrows():
                    gx  = row.get("gaussian_mean")
                    gxe = row.get("gaussian_mean_err")
                    if _nan(gx):           # fall back to smoothed position
                        gx, gxe = row.get("smoothed_x"), None
                    if not _nan(gx):
                        ax.axvline(gx, color=PEAK_COLOUR,
                                   linewidth=0.9, linestyle="--", alpha=0.75)
                        if not _nan(gxe):
                            ax.axvspan(gx - gxe, gx + gxe,
                                       color=PEAK_COLOUR, alpha=0.15, linewidth=0)

            # ── valley overlays ──────────────────────────────────────
            if show_valleys:
                vl_df = load_fit_results(results_dir, dig_id, ch, "valleys")
                for _, row in vl_df.iterrows():
                    gx  = row.get("gaussian_mean")
                    gxe = row.get("gaussian_mean_err")
                    if _nan(gx):
                        gx, gxe = row.get("smoothed_x"), None
                    if not _nan(gx):
                        ax.axvline(gx, color=VALLEY_COLOUR,
                                   linewidth=0.9, linestyle=":", alpha=0.75)
                        if not _nan(gxe):
                            ax.axvspan(gx - gxe, gx + gxe,
                                       color=VALLEY_COLOUR, alpha=0.13, linewidth=0)

            # ── labels ───────────────────────────────────────────────
            if row_idx == 0:
                ax.set_title(f"Ch {ch}", fontsize=8, fontweight="bold",
                             color="#C9D1D9", pad=5)
            if ch == 0:
                norm_tag = (f"\n(/{int(norm_factor):,})"
                            if norm_factor != 1.0 else "")
                ax.set_ylabel(f"DAQ {dig_id}{norm_tag}",
                              fontsize=7.5, color="#C9D1D9", labelpad=4)
            if row_idx == n_digs - 1:
                ax.set_xlabel("Amplitude bin", fontsize=6.5, color="#8B949E")

    # ── title + legend ───────────────────────────────────────────────
    fig.suptitle(
        "Amplitude Pulse-Height Spectra  --  All DAQs & Channels",
        fontsize=12, fontweight="bold", color="#E6EDF3", y=1.005,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=PEAK_COLOUR,   linewidth=1.3, linestyle="--",
                   label="Peak  (Gaussian mean +/- 1 sigma)"),
            Line2D([0], [0], color=VALLEY_COLOUR, linewidth=1.3, linestyle=":",
                   label="Valley (Gaussian mean +/- 1 sigma)"),
        ],
        loc="lower center", ncol=2, fontsize=8,
        facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9",
        bbox_to_anchor=(0.5, -0.022),
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(output_figure, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nFigure saved -> {output_figure}")
    plt.show()


#=======================================================================
# run
#=======================================================================
if __name__ == "__main__":
    plot_all_channels()