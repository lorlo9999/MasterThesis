"""
Reload saved phase-curve data and produce the stacked figure.
Usage:
    python plot_phase_curves.py                          # uses default data file
    python plot_phase_curves.py PhaseCurves/my_data.pkl  # custom data file
"""

import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data_path = sys.argv[1] if len(sys.argv) > 1 else "PhaseCurves/phase_curve_data.pkl"

with open(data_path, "rb") as fh:
    plot_data = pickle.load(fh)

n_plots = len(plot_data)
fig, axes = plt.subplots(
    n_plots, 1,
    figsize=(12, 3.2 * n_plots),
    constrained_layout=True,
)
if n_plots == 1:
    axes = [axes]

for ax, entry in zip(axes, plot_data):
    orbitals_h  = entry["orbitals_h"]
    binned_flux = entry["binned_flux"]
    rand        = entry["rand"]
    binned_err  = entry["binned_err"]
    binned_wl   = entry["binned_wl"]
    time        = entry["time"]
    name        = entry["name"]
    H           = entry["H"]
    iw          = entry["iw"]
    N_pc        = entry["N_pc"]

    ax.plot(orbitals_h, binned_flux, color="black", linewidth=0.9,
            label=f"{binned_wl:.2f} µm")
    ax.errorbar(orbitals_h, rand, yerr=binned_err,
                fmt="o", color="red", markersize=3, alpha=0.5, linewidth=0.6)
    ax.axhline(1, color="black", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Norm. Flux", fontsize=8)
    ax.set_title(
        f"{name}b  ·  H{H} IW{iw}  ·  {binned_wl:.2f} µm  "
        f"·  N={N_pc},  {time:.1f} d total",
        fontsize=8, loc="left",
    )
    ax.legend(fontsize=7, loc="upper right")
    ax.ticklabel_format(useOffset=False)
    ax.tick_params(labelsize=7)

axes[-1].set_xlabel("Time (hours from transit)", fontsize=10)

out_path = data_path.replace(".pkl", ".pdf").replace("_data", "")
fig.savefig(out_path, format="pdf", bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close(fig)
