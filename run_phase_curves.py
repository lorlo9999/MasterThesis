"""
Runs the phase-curve computation for every row in combined_nightside_results.csv
and saves a single vertically-stacked figure — one wavelength-binned phase curve
per row, no time binning.
"""

import math
import os
import pickle
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xarray as xr
import astropy
import astropy.constants

import taurex
taurex.log.disableLogging()

from taurex.cache import OpacityCache, CIACache
OpacityCache().clear_cache()
OpacityCache().set_opacity_path("./xsec")
CIACache().set_cia_path("./cia")

from taurex.planet import Planet
from taurex.stellar import PhoenixStar
from taurex.data.profiles.pressure import SimplePressureProfile
from taurex.temperature import TemperatureFile
from taurex.data.profiles.chemistry import TaurexChemistry
from taurex.chemistry import ConstantGas
from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution
from taurex.binning import FluxBinner

from explor.model import HotSpotPhaseCurveModel

# ── planet / star catalogue ───────────────────────────────────────────────────
planet_names         = ["HD3167","K2141","LHS1478","TOI431","TOI500","TOI561","TOI1416","TOI1807"]
planet_masses        = [4.73, 4.97, 2.33, 3.07, 1.42, 2.02, 3.48, 2.44]
planet_distances     = [0.018, 0.007, 0.018, 0.011, 0.012, 0.011, 0.019, 0.012]
planet_period        = [0.96, 0.28, 1.95, 0.49, 0.55, 0.45, 1.0, 0.55]
planet_period_hours  = [p * 24 for p in planet_period]
planet_radius        = [1.627, 1.510, 1.242, 1.277, 1.166, 1.397, 1.620, 1.496]
T_transit_hours      = [1.61, 0.94, 0.71, 1.24, 0.99, 1.31, 1.5, 0.98]
planet_transit       = [t * 3600 for t in T_transit_hours]
planet_impact        = [0.181, -0.01, 0.717, 0.34, 0.53, 0.14, 0.39, 0.489]
planet_eccentricity  = [0.05, 0.0, 0.0, 0.0, 0.06, 0.0, 0.0, 0.0]
planet_pericentre_long = [0.0, 90.0, 0.0, 0.0, 228.5, 0.0, 0.0, 90.0]

star_temperature = [5261.0, 4570.0, 3381.0, 4850.0, 4440.0, 5342.0, 4884.0, 4914.0]
star_radius      = [0.872, 0.681, 0.246, 0.731, 0.678, 0.856, 0.793, 0.746]
star_metallicity = [0.03, 0.0, -0.13, 0.2, 0.12, -0.4, 0.08, -0.04]
star_distance    = [47.28, 61.87, 18.22, 32.6, 47.39, 85.8, 55.01, 42.58]


# ── wavelength-binning helpers ────────────────────────────────────────────────
def bindown_single(w1, d1, w2, d2, output, noise, eclipses):
    wf = (w1 + w2) / 2
    df = (w2 + d2 / 2) - (w1 - d1 / 2)
    photogrid = np.array([[wf], [df]])
    fb = FluxBinner(photogrid[0], photogrid[1])
    wl, val, err, *_ = fb.bindown(output[0], output[1], error= noise / np.sqrt(eclipses))
    return w1, w2, wl[0], val[0], err[0], df


def make_next_level_points(results):
    assert len(results) % 2 == 0
    new_points = []
    for i in range(0, len(results), 2):
        lb, rb = results[i], results[i + 1]
        w1    = lb[2] - lb[5] / 2
        w2    = rb[2] + rb[5] / 2
        width = w2 - w1
        new_points.append((w1, width, w2, width))
    return new_points


def bindown_multiple(output, noise, eclipses, *new_points):
    return [bindown_single(w1, d1, w2, d2, output, noise, eclipses)
            for (w1, d1, w2, d2) in new_points]


# ── per-row computation ───────────────────────────────────────────────────────
def run_one(name, H, iw, eff, N_pc, S="40"):
    idx = planet_names.index(name)

    mass_jup   = planet_masses[idx] / 317.8
    radius_jup = planet_radius[idx] * (
        astropy.constants.R_earth.value / astropy.constants.R_jup.value
    )

    pl = Planet(
        planet_mass=mass_jup, planet_radius=radius_jup,
        planet_sma=planet_distances[idx], planet_distance=star_distance[idx],
        impact_param=planet_impact[idx], orbital_period=planet_period[idx],
        transit_time=planet_transit[idx],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = PhoenixStar(
            temperature=star_temperature[idx], radius=star_radius[idx],
            metallicity=star_metallicity[idx], distance=star_distance[idx],
            phoenix_path="Phoenix/",
        )

    pl._mid_time           = 0.0
    pl.pericentre_time     = 0.0
    pl.ascending_node_long = 0.0
    pl._eccentricity       = planet_eccentricity[idx]
    pl._pericentre_long    = planet_pericentre_long[idx]

    phase_window = 0.65
    _period      = planet_period[idx]
    sample_points = 100
    phases = np.linspace(-phase_window * _period, phase_window * _period, sample_points)

    # pressure / temperature profiles
    if name in ("K2141"):
        tp_path = f"./PLANETS/{name}/H{H}_IW{iw}_0001_S{S}_TP.csv"
    elif name in ("TOI431") and H == "10":
        tp_path = f"./PLANETS/{name}/H{H}_IW{iw}_0001_S40_TP.csv"
    elif name in ("TOI431") and H == "20":
        tp_path = f"./PLANETS/{name}/H{H}_IW{iw}_0001_S20_TP.csv"
    else:
        tp_path = f"./PLANETS/{name}/H{H}_IW{iw}_{eff}_TP.csv"

    data     = pd.read_csv(tp_path)
    pressure = data["Pressure (Pa)"].values
    p1 = SimplePressureProfile(
        nlayers=100,
        atm_min_pressure=float(np.min(pressure)),
        atm_max_pressure=float(np.max(pressure)),
    )
    p1.compute_pressure_profile()

    # column 1 = nightside T, column 2 = dayside T
    tp_night = TemperatureFile(tp_path, skiprows=1, temp_col=1, press_col=0,
                               temp_units="K", press_units="Pa", delimiter=",")
    tp_day   = TemperatureFile(tp_path, skiprows=1, temp_col=2, press_col=0,
                               temp_units="K", press_units="Pa", delimiter=",")
    temperatures = [tp_day, tp_day, tp_night]

    # chemistry from first .atm.nc found under PLANETS/{name}/
    atm_file = None
    for folder in os.listdir(f"./PLANETS/{name}/"):
        sim_path = os.path.join(f"./PLANETS/{name}/", folder)
        if not os.path.isdir(sim_path):
            continue
        for f in os.listdir(sim_path):
            if f.endswith("atm.nc"):
                atm_file = os.path.join(sim_path, f)
                break
        if atm_file:
            break

    ds    = xr.open_dataset(atm_file)
    gases = [m.decode().strip() for m in ds["gases"].values]
    vmr   = np.array(ds["x_gas"])

    def get_vmr(mol):
        return float(vmr[:, gases.index(mol)][0]) if mol in gases else 0.0

    H2O_x = get_vmr("H2O"); CO2_x = get_vmr("CO2"); CH4_x = get_vmr("CH4")
    CO_x  = get_vmr("CO");  NH3_x = get_vmr("NH3"); N2_x  = get_vmr("N2")
    SO2_x = get_vmr("SO2"); S2_x  = get_vmr("S2");  O2_x  = get_vmr("O2")
    H2_x  = get_vmr("H2");  H2S_x = get_vmr("H2S")

    total = H2O_x+CO2_x+CH4_x+CO_x+NH3_x+N2_x+SO2_x+S2_x+O2_x+H2_x+H2S_x
    if total > 1:
        H2O_x/=total; CO2_x/=total; CH4_x/=total; CO_x/=total; NH3_x/=total
        N2_x/=total;  SO2_x/=total; S2_x/=total;  O2_x/=total
        H2_x/=total;  H2S_x/=total

    chemistry = TaurexChemistry(fill_gases=["N2"])
    (chemistry
     .addGas(ConstantGas(molecule_name="NH3", mix_ratio=NH3_x))
     .addGas(ConstantGas(molecule_name="CO2", mix_ratio=CO2_x))
     .addGas(ConstantGas(molecule_name="H2O", mix_ratio=H2O_x))
     .addGas(ConstantGas(molecule_name="CH4", mix_ratio=CH4_x))
     .addGas(ConstantGas(molecule_name="CO",  mix_ratio=CO_x))
     .addGas(ConstantGas(molecule_name="SO2", mix_ratio=SO2_x))
     .addGas(ConstantGas(molecule_name="S2",  mix_ratio=S2_x))
     .addGas(ConstantGas(molecule_name="O2",  mix_ratio=O2_x))
     .addGas(ConstantGas(molecule_name="H2",  mix_ratio=H2_x))
     .addGas(ConstantGas(molecule_name="H2S", mix_ratio=H2S_x)))

    cia = ["CO2-CO2", "CO2-H2", "CO2-H2O", "H2-H2", "O2-CO2"]
    ctribs = [
        [AbsorptionContribution(), RayleighContribution(), CIAContribution(cia_pairs=cia)],
        [AbsorptionContribution(), RayleighContribution(), CIAContribution(cia_pairs=cia)],
        [AbsorptionContribution(), RayleighContribution(), CIAContribution(cia_pairs=cia)],
    ]

    hs = HotSpotPhaseCurveModel(
        phases=phases,
        temperature_profiles=temperatures,
        chemistry=[chemistry, chemistry, chemistry],
        nlayers=[70, 70, 70],
        pressure_profile=[p1, p1, p1],
        planet=pl, star=st,
        observation=None, contributions=ctribs,
        alpha_hs=45.0, delta_hs=-40.0, ngauss=40,
        use_directimage=False, use_cuda=False, use_orbitals=True,
        temperature_constraints=10, res_grid=[400, 0.2, 18.5],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hs.build()
        o = hs.model()

    # bin to ARIEL wavelength grid
    ariel = pd.read_csv(f"./ARIEL/arielrad_{name}/tier2.csv", skiprows=6)
    wave          = ariel["Wavelength [um]"].values
    error_w_floor = ariel["Noise on Transit Floor [ppm]"].values * 1e-6
    #call it error with floor to avoid changing variables in the rest of the script
    error_w_floor = ariel['Total Noise [ppm]'].values*1e-6

    fb2  = FluxBinner(wave)
    flux = np.zeros((len(hs._orbitals), len(wave)))
    for i in range(len(hs._orbitals)):
        _, flux[i, :], _, _ = fb2.bindown(
            10000 / hs.wls[0][::-1], np.array(o[1])[::-1, i]
        )

    # three-level wavelength binning → single broadband point per phase
    new_points = [
        (4.305457,    0.28225, 4.5973,      0.3014),
        (4.908865,    0.322,   5.24157,     0.344),
        (5.59683967,  0.367,   5.97618103,  0.391),
        (6.38,        0.418,   6.814,       0.447),
    ]

    time         = N_pc * planet_period[idx]
    # time per phase point, in hours
    dtsampled    = np.mean(hs._orbitals[1:] - hs._orbitals[:-1]) * 24
    # factor to scale noise by, considers stacked phase curves + scales by actual hours per point
    eff_eclipses = N_pc * dtsampled  # σ₁ is per hour; scale by actual hours per point

    binned_flux = np.zeros(len(hs._orbitals))
    results_l3  = None
    for i in range(len(hs._orbitals)):
        out_i      = (wave, flux[i, :])
        results_l1 = bindown_multiple(out_i, error_w_floor, eff_eclipses, *new_points)
        results_l2 = bindown_multiple(out_i, error_w_floor, eff_eclipses,
                                      *make_next_level_points(results_l1))
        results_l3 = bindown_multiple(out_i, error_w_floor, eff_eclipses,
                                      *make_next_level_points(results_l2))
        binned_flux[i] = results_l3[0][3]

    binned_err = results_l3[0][4]
    binned_wl  = results_l3[0][2]

    # noise draw — retry until nightside bottom > 1
    _quarter      = planet_period[idx] / 4
    _near_transit = np.abs(hs._orbitals) < _quarter
    transit_center_idx = np.where(_near_transit)[0][
        np.argmin(binned_flux[_near_transit])
    ]
    _transit_half  = (T_transit_hours[idx] / 2) / 24
    _dt            = np.mean(np.diff(hs._orbitals))
    ingress_idx    = transit_center_idx - int(round(_transit_half / _dt)) - 1
    before_ingress = np.arange(ingress_idx - 2, ingress_idx + 2)
    _err_night     = binned_err / np.sqrt(4)

    best_rand, best_margin = None, -np.inf
    for _ in range(10_000):
        rand = np.random.normal(binned_flux, binned_err)
        r1, r2, r3, r4 = rand[before_ingress]
        margin = ((r1 + r2) / 2 + (r3 + r4) / 2) / 2 - _err_night - 1
        if margin > best_margin:
            best_margin, best_rand = margin, rand
        if margin > 0:
            break
    rand = best_rand

    return hs._orbitals * 24, binned_flux, rand, binned_err, binned_wl, time


# ── main: collect results and build stacked figure ────────────────────────────
df          = pd.read_csv("combined_nightside_results.csv", index_col=0)
rows_to_run = [(i, row) for i, row in df.iterrows()]
n_plots     = len(rows_to_run)

fig, axes = plt.subplots(
    n_plots, 1,
    figsize=(12, 3.2 * n_plots),
    constrained_layout=True,
)
if n_plots == 1:
    axes = [axes]

os.makedirs("PhaseCurves", exist_ok=True)

total      = len(rows_to_run)
plot_data  = []   # accumulates one dict per successful run

for ax_idx, (i, row) in enumerate(rows_to_run):
    name  = row["Planet"]
    H     = row["Hydrogen Inventory [H oceans]"].lstrip("H")
    iw    = row["Redox State"].lstrip("IW")
    eff   = "00001"
    N_pc  = math.floor((row["Number of observations"] + row["Observations with bb fit"]) / 2)
    S     = "40"

    label = f"{name}_H{H}_IW{iw}_{eff}"
    print(f"[{ax_idx+1}/{total}] Running {label}  (N_pc={N_pc})")

    try:
        orbitals_h, binned_flux, rand, binned_err, binned_wl, time = \
            run_one(name, H, iw, eff, N_pc, S=S)
    except Exception as e:
        print(f"  FAILED: {label}\n  {e}")
        axes[ax_idx].set_visible(False)
        continue

    plot_data.append(dict(
        label=label, name=name, H=H, iw=iw, eff=eff, N_pc=N_pc,
        orbitals_h=orbitals_h,
        binned_flux=binned_flux,
        rand=rand,
        binned_err=float(binned_err),
        binned_wl=float(binned_wl),
        time=float(time),
    ))

    ax = axes[ax_idx]
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
    print(f"  Done: {label}")

axes[-1].set_xlabel("Time (hours from transit)", fontsize=10)

data_path = "PhaseCurves/phase_curve_data.pkl"
with open(data_path, "wb") as fh:
    pickle.dump(plot_data, fh)
print(f"Saved plot data: {data_path}")

out_path = "PhaseCurves/all_phase_curves.pdf"
fig.savefig(out_path, format="pdf", bbox_inches="tight")
print(f"Saved figure:    {out_path}")
plt.close(fig)
