import pandas as pd
import numpy as np

# === Stellar parameters (one row per star) ===
stars = pd.DataFrame({
    "Planet":          ["HD 3167b", "K2-141b", "LHS 1478b", "TOI-431b", "TOI-500b", "TOI-561b", "TOI-1416b", "TOI-1807b"],
    "T_eff [K]":       [5261, 4570, 3381, 4850, 4440, 5342, 4884, 4914],
    "R_star [R_sun]":  [0.872, 0.681, 0.246, 0.731, 0.678, 0.856, 0.793, 0.746],
    "[Fe/H] [dex]":    [0.03, 0.00, -0.13, 0.20, 0.12, -0.40, 0.08, -0.04],
    "log_g [cgs]":     [4.5, 4.6, 4.9, 4.6, 4.6, 4.5, 4.5, 4.6],
    "Age [Gyr]":       [10.2, 6.3, 5.6, 5.1, 5.0, 11.0, 6.9, 0.3],
    "d [pc]":          [47.28, 61.87, 18.22, 32.60, 47.39, 85.80, 55.01, 42.58],
})

# === Planetary parameters (one row per planet) ===
planets = pd.DataFrame({
    "Planet":          ["HD 3167b", "K2-141b", "LHS 1478b", "TOI-431b", "TOI-500b", "TOI-561b", "TOI-1416b", "TOI-1807b"],
    "M_p [M_earth]":   [4.73, 4.97, 2.33, 3.07, 1.42, 2.02, 3.48, 2.44],
    "R_p [R_earth]":   [1.627, 1.510, 1.242, 1.277, 1.166, 1.397, 1.620, 1.496],
    "P [days]":        [0.96, 0.28, 1.95, 0.49, 0.55, 0.45, 1.00, 0.55],
    "a [AU]":          [0.018, 0.007, 0.018, 0.011, 0.012, 0.011, 0.019, 0.012],
    "T_14 [hours]":    [1.61, 0.94, 0.71, 1.24, 0.99, 1.31, 1.50, 0.98],
    "b":               [0.181, -0.01, 0.717, 0.34, 0.53, 0.14, 0.39, 0.489],
    "e":               [0.05, 0.0, 0.0, 0.0, 0.06, 0.0, 0.0, 0.0],
    "omega [deg]":     [0.0, 90.0, 0.0, 0.0, 228.5, 0.0, 0.0, 90.0],
})

# === Observational results (multiple rows per planet) ===
obs = pd.DataFrame({
    "Case #": [34, 46, 88, 124, 1, 11, 34, 38, 26, 22, 34, 82, 46, 42, 46],
    "Planet": [
        "HD 3167b", "HD 3167b", "K2-141b", "K2-141b", "LHS 1478b", "LHS 1478b",
        "TOI-1416b", "TOI-1416b", "TOI-1807b", "TOI-1807b", "TOI-431b", "TOI-431b",
        "TOI-500b", "TOI-500b", "TOI-561b",
    ],
    "Atmospheric Components": [
        "SO2, H2O", "CO2, N2", "SO2, H2O", "SO2, S2", "H2O, S2",
        "CO2, H2O", "CO2, SO2", "CO, N2", "CO, N2", "SO2, H2O",
        "SO2, S2", "SO2, H2O", "H2O, SO2", "H2O, H2", "H2O, O2",
    ],
    "P_surf [bar]":           [10.90, 301.89, 169.22, 16.03, 317.21, 2693.22, 55.97, 437.02, 153.47, 13.57, 180.56, 256.23, 217.26, 20.38, 45.08],
    "H Inventory [H_oceans]": [10, 20, 20, 30, 5, 5, 10, 20, 10, 5, 10, 20, 20, 20, 20],
    "Redox State [IW]":       [4, 4, 4, 2, 0, 4, 4, 0, 0, 4, 4, 4, 4, 2, 4],
    "MMW [g/mol]":            [53.05, 41.45, 57.86, 55.35, 34.93, 36.81, 46.02, 26.35, 25.39, 54.36, 60.00, 54.14, 40.00, 17.39, 25.32],
    "T_day [K]":              [2495, 2567, 3023, 2935, 1218, 1303, 2254, 2390, 2571, 2441, 2717, 2850, 2803, 2734, 3193],
    "T_night [K]":            [1095, 1234, 1337, 1242, 335, 408, 1023, 1189, 1214, 1048, 1235, 1309, 1113, 1212, 1518],
    "N_obs":                  [15, 13, 8, 8, 10000, 10000, 49, 35, 21, 32, 9, 8, 36, 33, 35],
    "N_bb":                   [18, 15, 8, 8, 9850, 9850, 53, 37, 21, 32, 10, 10, 36, 38, 34],
    "T_obs [hours]":          [345.60, 299.52, 53.76, 53.76, 468000.00, 468000.00, 1176.00, 840.00, 277.20, 422.40, 105.84, 94.08, 475.20, 435.60, 378.00],
    "Retrieval Evidence":     [2.4, 1.7, 2.8, 2.5, np.nan, np.nan, 4.9, 1.9, 1.7, 4.8, 2.4, 2.1, 2.7, 2.1, 2.0],
    "A_obs":          [0.897, 0.895, 0.896, 0.893, np.nan, np.nan, 0.911, 0.904, 0.894, 0.906, 0.902, 0.896, 0.905, 0.912, 0.863],
    "σ_obs":          [0.196, 0.190, 0.182, 0.180, np.nan, np.nan, 0.201, 0.199, 0.203, 0.202, 0.197, 0.188, 0.202, 0.197, 0.203],
    "N_pc":          [5, 6, 3, 3,np.nan, np.nan, 15, 14, 9, 10, 3, 3, 12, 10, 21],
    "A_obs_AIRS":     [0.870, 0.895, 0.862, 0.858, np.nan, np.nan, 0.885, 0.873, 0.862,0.879, 0.873, 0.866, 0.874, 0.881, 0.827],
    "σ_obs_AIRS":     [0.188, 0.192, 0.185, 0.183, np.nan, np.nan, 0.201, 0.204, 0.199, 0.200, 0.184, 0.202, 0.199, 0.202, 0.202],
    "N_pc_AIRS":     [6, 6, 4, 4, np.nan, np.nan, 15, 14, 10, 11, 4,3, 14, 11, 26],
})

# === Merge into supertable ===
df = obs.merge(planets, on="Planet", how="left").merge(stars, on="Planet", how="left")

# === Gaussian scaling: N_eclipses / T_eclipses for 3-sigma detection ===
# σ ∝ √N  =>  N_eclipses = ceil( N_obs × (3/σ)² )
# T_eclipses [hours] = P [days]×24 + (N_eclipses − 2) × T_14 [hours]
df["N_eclipses"] = df["N_obs"]+1
df["T_eclipses [hours]"] = df["P [days]"] * 24 + (df["N_eclipses"] - 2) * df["T_14 [hours]"]

# === Free evidence from phase curves (display only — not used for scaling above) ===
# Each phase curve yields 2 secondary eclipses; σ ∝ √N  =>  σ_free = σ × √2
#df["Free Retrieval Evidence"] = df["Retrieval Evidence"] * np.sqrt(2)
#replace displayed evidence column with free evidence
#df["Retrieval Evidence"] = df["Free Retrieval Evidence"]

# === Reorder columns: Planet | Planetary Params | Stellar Params | Observational Results ===
col_order = [
    "Planet",
    # Planetary parameters
    "M_p [M_earth]", "R_p [R_earth]", "P [days]", "a [AU]", "T_14 [hours]", "b", "e", "omega [deg]",
    # Stellar parameters
    "T_eff [K]", "R_star [R_sun]", "[Fe/H] [dex]", "log_g [cgs]", "Age [Gyr]", "d [pc]",
    # Observational results
    "Case #", "Atmospheric Components", "P_surf [bar]", "H Inventory [H_oceans]",
    "Redox State [IW]", "MMW [g/mol]", "T_day [K]", "T_night [K]",
    "N_obs", "N_bb", "T_obs [hours]", "Retrieval Evidence",
    "N_eclipses", "T_eclipses [hours]"
]
df = df[col_order]

# Sort by planet name then case number
df = df.sort_values(["Planet", "Case #"]).reset_index(drop=True)
