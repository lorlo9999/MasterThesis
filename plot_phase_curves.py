"""
Reload saved phase-curve data and produce the stacked figure.
Usage:
    python plot_phase_curves.py                          # uses default data file
    python plot_phase_curves.py PhaseCurves/my_data.pkl  # custom data file
"""

import math
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supertable_code import obs

# map run-script planet keys → supertable planet names
_NAME_MAP = {
    "HD3167":  "HD 3167b",
    "K2141":   "K2-141b",
    "LHS1478": "LHS 1478b",
    "TOI431":  "TOI-431b",
    "TOI500":  "TOI-500b",
    "TOI561":  "TOI-561b",
    "TOI1416": "TOI-1416b",
    "TOI1807": "TOI-1807b",
}

def lookup_case(name, H, iw):
    """Return the Case # row from obs matching planet / H inventory / redox state."""
    full_name = _NAME_MAP.get(name, name)
    mask = (
        (obs["Planet"] == full_name) &
        (obs["H Inventory [H_oceans]"] == int(H)) &
        (obs["Redox State [IW]"] == int(iw))
    )
    rows = obs[mask]
    if rows.empty:
        return None, full_name
    row = rows.iloc[0]
    return int(row["Case #"]), full_name


data_path = sys.argv[1] if len(sys.argv) > 1 else "PhaseCurves/phase_curve_data.pkl"

with open(data_path, "rb") as fh:
    plot_data = pickle.load(fh)

PLOTS_PER_PAGE = 7
# A4 portrait in inches
PAGE_W, PAGE_H = 8.27, 11.69

def draw_entry(ax, entry):
    orbitals_h  = entry["orbitals_h"]
    binned_flux = entry["binned_flux"]
    rand        = entry["rand"]
    binned_err  = entry["binned_err"]
    time        = entry["time"]
    name        = entry["name"]
    H           = entry["H"]
    iw          = entry["iw"]
    N_pc        = entry["N_pc"]
    time_hours  = time * 24

    case_num, full_name = lookup_case(name, H, iw)
    case_str = f"Case #{case_num}" if case_num is not None else f"H{H} IW{iw}"

    ax.plot(orbitals_h, binned_flux, color="black", linewidth=0.9)
    ax.errorbar(orbitals_h, rand, yerr=binned_err,
                fmt="o", color="red", markersize=2, alpha=0.5, linewidth=0.8)
    ax.axhline(1, color="black", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Norm. Flux", fontsize=11)
    ax.set_title(
        f"{full_name}  ·  {case_str}  ·  N={N_pc},  {time_hours:.0f} h",
        fontsize=11, loc="left",
    )
    ax.ticklabel_format(useOffset=False)
    ax.tick_params(labelsize=10)


base_path = data_path.replace(".pkl", "").replace("_data", "")
n_pages   = math.ceil(len(plot_data) / PLOTS_PER_PAGE)

for page in range(n_pages):
    chunk = plot_data[page * PLOTS_PER_PAGE : (page + 1) * PLOTS_PER_PAGE]
    n     = len(chunk)

    fig, axes = plt.subplots(
        n, 1,
        figsize=(PAGE_W, PAGE_H),
        constrained_layout=True,
    )
    if n == 1:
        axes = [axes]

    for ax, entry in zip(axes, chunk):
        draw_entry(ax, entry)

    axes[-1].set_xlabel("Time (hours from transit)", fontsize=12)

    out_path = f"{base_path}_p{page + 1}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

print(f"Done ({n_pages} file{'s' if n_pages > 1 else ''} total)")

# ── grid figure: 7 rows (planets) × 2 columns (cases), shared y per row ──────
from collections import defaultdict

PLANET_ORDER = [
    "HD 3167b", "K2-141b", "LHS 1478b", "TOI-431b", "TOI-500b",
    "TOI-561b", "TOI-1416b", "TOI-1807b",
]

planet_groups = defaultdict(list)
for entry in plot_data:
    case_num, full_name = lookup_case(entry["name"], entry["H"], entry["iw"])
    planet_groups[full_name].append((case_num, entry))

# sort each planet's cases by case number
for key in planet_groups:
    planet_groups[key].sort(key=lambda x: x[0] or 0)

rows = [p for p in PLANET_ORDER if p in planet_groups]
n_rows = len(rows)

fig, axes = plt.subplots(
    n_rows, 2,
    sharey="row",
    figsize=(PAGE_W, PAGE_H),
    constrained_layout=True,
)

for row_idx, full_name in enumerate(rows):
    cases = planet_groups[full_name]
    for col_idx in range(2):
        ax = axes[row_idx, col_idx]
        if col_idx >= len(cases):
            ax.set_visible(False)
            continue

        case_num, entry = cases[col_idx]
        orbitals_h  = entry["orbitals_h"]
        binned_flux = entry["binned_flux"]
        rand        = entry["rand"]
        binned_err  = entry["binned_err"]
        time_hours  = entry["time"] * 24
        N_pc        = entry["N_pc"]
        case_str    = f"Case #{case_num}" if case_num is not None else f"H{entry['H']} IW{entry['iw']}"

        ax.plot(orbitals_h, binned_flux, color="black", linewidth=0.9)
        ax.errorbar(orbitals_h, rand, yerr=binned_err,
                    fmt="o", color="red", markersize=2, alpha=0.5, linewidth=0.8)
        ax.axhline(1, color="black", linestyle="--", linewidth=0.5)
        ax.set_title(
            f"{full_name}  ·  {case_str}  ·  N={N_pc},  {time_hours:.0f} h"
            if col_idx == 0 else
            f"{case_str}  ·  N={N_pc},  {time_hours:.0f} h",
            fontsize=9, loc="left",
        )
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(labelsize=8)
        if col_idx == 0:
            ax.set_ylabel("Norm. Flux", fontsize=9)

for col_idx in range(2):
    axes[-1, col_idx].set_xlabel("Time (hours from transit)", fontsize=10)

grid_path = f"{base_path}_grid.pdf"
fig.savefig(grid_path, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {grid_path}")

# ── standalone example figure: K2-141b Case #88 ───────────────────────────────
_target = next(
    (e for e in plot_data if e["name"] == "K2141" and e["H"] == "20" and e["iw"] == "4"),
    None,
)
if _target is not None:
    fig, ax = plt.subplots(figsize=(10, 4))

    orbitals_h  = _target["orbitals_h"]
    binned_flux = _target["binned_flux"]
    rand        = _target["rand"]
    binned_err  = _target["binned_err"]
    time_hours  = _target["time"] * 24
    N_pc        = _target["N_pc"]
    binned_wl   = _target["binned_wl"]

    ax.plot(orbitals_h, binned_flux, color="black", linewidth=1.2,
            label=f"Model  ({binned_wl:.2f} µm)")
    ax.errorbar(orbitals_h, rand, yerr=binned_err,
                fmt="o", color="red", markersize=4, alpha=0.7, linewidth=1.0,
                label=f"Simulated obs.  (N={N_pc}, {time_hours:.0f} h)")
    ax.axhline(1, color="black", linestyle="--", linewidth=0.7, label="Stellar baseline")

    ax.set_xlabel("Time from transit centre (hours)", fontsize=13)
    ax.set_ylabel("Normalised flux", fontsize=13)
    ax.set_title("K2-141b  ·  Case #88", fontsize=13, loc="left")
    ax.legend(fontsize=10, frameon=False)
    ax.ticklabel_format(useOffset=False)
    ax.tick_params(labelsize=11)

    example_path = f"{base_path}_k2141_case88.pdf"
    fig.savefig(example_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {example_path}")
else:
    print("K2-141b Case #88 not found in data — skipping example figure.")
