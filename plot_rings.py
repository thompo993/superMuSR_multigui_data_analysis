#=======================================================================
# plot_ring_overlays.py
# Plots pulse-height spectra as overlays grouped by detector "ring".
#
# Ring assignments:
#   ring0 : DAQ0  ch 0-3      ring1 : DAQ0  ch 4-7
#   ring2 : DAQ1  ch 0-3      ring3 : DAQ1  ch 4-7
#   ring4 : DAQ2  ch 0-3      ring5 : DAQ2  ch 4-7
#   ring6 : DAQ3  ch 0-3      ring7 : DAQ3  ch 4-7
#
# Each subplot shows all channels in a ring overlaid on the same axes,
# with optional peak / valley markers inherited from the analysis CSVs.
#=======================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

#=======================================================================
# inputs — edit these to match your setup
#=======================================================================
RESULTS_DIR   = r"./data"
RAW_DATA_DIR  = r"./test_parent_dir_1"
OUTPUT_FIGURE = r"./phs_ring_overlays.png"

NORMALISE    = True     # must match what was used in the analysis script
LOG_SCALE    = True
SHOW_PEAKS   = True
SHOW_VALLEYS = True

# Layout: how many ring panels per row in the figure
RINGS_PER_ROW = 4

# Colours for the 4 channels within each ring (indices 0-3 within the ring)
RING_CH_COLOURS = [
    "#ffff00",   # first channel in ring
    "#ff6b35",   # second
    "#00e5ff",   # third
    "#b5e853",   # fourth
]

PEAK_COLOUR   = "#E63946"
VALLEY_COLOUR = "#457B9D"

# Ring definitions: list of (dag_id, channel) tuples per ring index
RING_DEFINITIONS = {
    0: [(0, 0), (0, 1), (0, 2), (0, 3)],
    1: [(0, 4), (0, 5), (0, 6), (0, 7)]
}
# RING_DEFINITIONS = {
#     0: [(0, 0), (0, 1), (0, 2), (0, 3)],
#     1: [(0, 4), (0, 5), (0, 6), (0, 7)],
#     2: [(1, 0), (1, 1), (1, 2), (1, 3)],
#     3: [(1, 4), (1, 5), (1, 6), (1, 7)],
#     4: [(2, 0), (2, 1), (2, 2), (2, 3)],
#     5: [(2, 4), (2, 5), (2, 6), (2, 7)],
#     6: [(3, 0), (3, 1), (3, 2), (3, 3)],
#     7: [(3, 4), (3, 5), (3, 6), (3, 7)],
# }

#=======================================================================
# helpers  (ported / simplified from plot_phs_results.py)
#=======================================================================

def _load_amplitude_csv(filepath):
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


def load_spectrum(raw_data_dir, dig_id):
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
        return _load_amplitude_csv(candidates[0])
    except Exception as exc:
        print(f"  Could not load spectrum for DAQ {dig_id}: {exc}")
        return None


def load_fit_results(results_dir, dig_id, ch, feature_type):
    path = Path(results_dir) / f"daq{dig_id}_ch{ch}_{feature_type}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_norm_factor(results_dir, dig_id):
    for ch in range(8):
        pk = load_fit_results(results_dir, dig_id, ch, "peaks")
        if not pk.empty and "norm_factor" in pk.columns:
            return float(pk["norm_factor"].iloc[0])
    return 1.0


def _nan(val):
    try:
        return val is None or (isinstance(val, float) and np.isnan(val))
    except Exception:
        return True


def _first_nonzero(arr):
    nz = np.where(arr > 0)[0]
    return int(nz[0]) if len(nz) else 0


def _active_xlim(counts, margin_frac=0.05):
    if not np.any(counts > 0):
        return 0, len(counts) - 1
    start = _first_nonzero(counts)
    end   = int(np.where(counts > 0)[0][-1]) + 1
    span  = max(end - start, 1)
    margin = max(int(span * margin_frac), 2)
    return max(0, start - margin), min(len(counts) - 1, end + margin)


#=======================================================================
# main plotting function
#=======================================================================
def plot_ring_overlays(
        results_dir   = RESULTS_DIR,
        raw_data_dir  = RAW_DATA_DIR,
        output_figure = OUTPUT_FIGURE,
        log_scale     = LOG_SCALE,
        show_peaks    = SHOW_PEAKS,
        show_valleys  = SHOW_VALLEYS,
        normalise     = NORMALISE,
        rings_per_row = RINGS_PER_ROW,
        ring_defs     = RING_DEFINITIONS,
):
    results_dir   = Path(results_dir)
    raw_data_dir  = Path(raw_data_dir)
    output_figure = Path(output_figure)

    n_rings = len(ring_defs)
    n_cols  = min(rings_per_row, n_rings)
    n_rows  = int(np.ceil(n_rings / n_cols))

    # Pre-load spectra and norm factors per DAQ (cache to avoid re-reading)
    daq_ids = sorted({dig for members in ring_defs.values() for dig, _ in members})
    spec_cache = {}
    norm_cache = {}
    for did in daq_ids:
        spec_cache[did] = load_spectrum(raw_data_dir, did)
        norm_cache[did] = get_norm_factor(results_dir, did)
        if spec_cache[did] is None:
            print(f"  Warning: no spectrum found for DAQ {did}")

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(n_cols * 4.2, n_rows * 3.2),
        sharex=False, sharey=False,
    )
    fig.patch.set_facecolor("#0F1117")

    # Normalise axes shape to 2-D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for ring_idx in range(n_rows * n_cols):
        row, col = divmod(ring_idx, n_cols)
        ax = axes[row, col]

        # Style panel
        ax.set_facecolor("#161B22")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
            spine.set_linewidth(0.8)
        ax.tick_params(colors="#8B949E", labelsize=6, length=3, width=0.6)

        # Hide unused panels (if n_rings is not a multiple of n_cols)
        if ring_idx >= n_rings:
            ax.set_visible(False)
            continue

        members = ring_defs[ring_idx]   # list of (dig_id, ch)
        ax.set_title(f"Ring {ring_idx}",
                     fontsize=9, fontweight="bold",
                     color="#C9D1D9", pad=5)

        # Track x-limits across all channels in the ring so we can union them
        all_x_lo, all_x_hi = [], []
        legend_lines = []   # for the in-panel legend

        for local_idx, (dig_id, ch) in enumerate(members):
            colour = RING_CH_COLOURS[local_idx % len(RING_CH_COLOURS)]
            label  = f"DAQ{dig_id} Ch{ch}"

            spec_df     = spec_cache.get(dig_id)
            norm_factor = norm_cache.get(dig_id, 1.0)

            plotted = False
            if spec_df is not None and ch < spec_df.shape[1]:
                raw    = spec_df.iloc[:, ch].values.astype(float)
                counts = raw / norm_factor if (normalise and norm_factor != 1.0) else raw
                bins   = np.arange(len(counts), dtype=float)

                x_lo, x_hi = _active_xlim(counts)
                all_x_lo.append(x_lo)
                all_x_hi.append(x_hi)

                display = np.where(counts > 0, counts, np.nan)
                line, = ax.plot([], [], color=colour, linewidth=0.9,
                                label=label, alpha=0.9)   # dummy for legend
                ax.step(bins, display, where="mid",
                        color=colour, linewidth=0.85, alpha=0.9)
                ax.fill_between(bins, display, step="mid",
                                color=colour, alpha=0.10)
                legend_lines.append(
                    Line2D([0], [0], color=colour, linewidth=1.4, label=label)
                )
                plotted = True

            # ── peaks ──────────────────────────────────────────────
            if show_peaks:
                pk_df = load_fit_results(results_dir, dig_id, ch, "peaks")
                for _, r in pk_df.iterrows():
                    gx  = r.get("gaussian_mean")
                    gxe = r.get("gaussian_mean_err")
                    if _nan(gx):
                        gx, gxe = r.get("smoothed_x"), None
                    if not _nan(gx):
                        ax.axvline(gx, color=PEAK_COLOUR,
                                   linewidth=0.8, linestyle="--", alpha=0.75)
                        if not _nan(gxe):
                            ax.axvspan(gx - gxe, gx + gxe,
                                       color=PEAK_COLOUR, alpha=0.10, linewidth=0)

            # ── valleys ────────────────────────────────────────────
            if show_valleys:
                vl_df = load_fit_results(results_dir, dig_id, ch, "valleys")
                for _, r in vl_df.iterrows():
                    gx  = r.get("gaussian_mean")
                    gxe = r.get("gaussian_mean_err")
                    if _nan(gx):
                        gx, gxe = r.get("smoothed_x"), None
                    if not _nan(gx):
                        ax.axvline(gx, color=VALLEY_COLOUR,
                                   linewidth=0.8, linestyle=":", alpha=0.75)
                        if not _nan(gxe):
                            ax.axvspan(gx - gxe, gx + gxe,
                                       color=VALLEY_COLOUR, alpha=0.08, linewidth=0)

            if not plotted:
                ax.text(0.5, 0.5, f"no data\n{label}",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=6.5,
                        color="#484F58", fontstyle="italic")

        # ── apply unioned x-limits across ring ───────────────────
        if all_x_lo:
            ax.set_xlim(min(all_x_lo), max(all_x_hi))

        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=4))
            ax.yaxis.set_minor_locator(ticker.NullLocator())

        # ── per-panel legend (channel labels) ────────────────────
        if legend_lines:
            leg = ax.legend(
                handles=legend_lines,
                fontsize=5.5, loc="upper right",
                facecolor="#0F1117", edgecolor="#30363D",
                labelcolor="#C9D1D9", framealpha=0.85,
                handlelength=1.4, handletextpad=0.5,
                borderpad=0.5,
            )

        # axis labels
        if col == 0:
            y_label = "Counts (normalised)" if normalise else "Counts"
            ax.set_ylabel(y_label, fontsize=7, color="#C9D1D9", labelpad=4)
        if row == n_rows - 1:
            ax.set_xlabel("Amplitude bin", fontsize=6.5, color="#8B949E")

    # ── figure-level title + legend ──────────────────────────────
    fig.suptitle(
        "Pulse-Height Spectra  —  Ring Overlays",
        fontsize=13, fontweight="bold", color="#E6EDF3",
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=PEAK_COLOUR,   linewidth=1.3, linestyle="--",
                   label="Peak  (Gaussian mean ± 1σ)"),
            Line2D([0], [0], color=VALLEY_COLOUR, linewidth=1.3, linestyle=":",
                   label="Valley (Gaussian mean ± 1σ)"),
        ],
        loc="lower center", ncol=2, fontsize=8,
        facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9",
        bbox_to_anchor=(0.5, -0.022),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_figure, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nFigure saved → {output_figure}")
    plt.show()


#=======================================================================
# run
#=======================================================================
if __name__ == "__main__":
    plot_ring_overlays()