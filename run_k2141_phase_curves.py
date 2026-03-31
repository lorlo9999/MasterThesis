"""
Runs phase_curve.ipynb for every row in combined_nightside_results.csv.
N_pc is set to floor((Number of observations + Observations with bb fit) / 2).
"""

import math
import pandas as pd
import papermill as pm

df = pd.read_csv("combined_nightside_results.csv", index_col=0)

total = len(df)
for run_num, (i, row) in enumerate(df.iterrows(), start=1):
    name = row["Planet"]

    if name != "K2141":
        print(f"\n[{run_num}/{total}] Skipping {name}")
        continue

    H = row["Hydrogen Inventory [H oceans]"].lstrip("H")   # "H10" -> "10"
    iw = row["Redox State"].lstrip("IW")                   # "IW4" -> "4"
    eff = "0001"
    n_obs = row["Number of observations"]
    n_bb  = row["Observations with bb fit"]
    N_pc  = math.floor((n_obs + n_bb) / 2)

    label = f"{name}_H{H}_IW{iw}_{eff}_S40"
    print(f"\n[{run_num}/{total}] Running {label}  (N_pc={N_pc})")

    try:
        pm.execute_notebook(
            "phase_curve.ipynb",
            f"PhaseCurves/{label}_output.ipynb",
            parameters=dict(
                name=name,
                H=H,
                iw=iw,
                eff=eff,
                S=40,
                N_pc=N_pc,
            ),
            kernel_name="python3",
        )
        print(f"  Done: {label}")
    except Exception as e:
        print(f"  FAILED: {label}\n  {e}")
        continue

print("\nAll runs complete.")