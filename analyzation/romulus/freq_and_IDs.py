import glob
import os
import re
import pandas as pd
import tangos
import matplotlib.pyplot as plt
from collections import defaultdict

################################################################################
# User settings
################################################################################
CATALOG_DIR = "/scratch/stlock/tripleAGNs/catalogs/1e41lum/catalogue_15kpc_1e41lum"
OUTPUT_CSV = "tripleAGN_rtgen_targets.csv"
OUTPUT_PLOT = "tripleAGN_frequency_vs_redshift.png"

################################################################################
# Tangos setup
################################################################################
sim = tangos.get_simulation("cosmo25")

################################################################################
# Containers
################################################################################
rows_out = []
freq_by_z = {}

################################################################################
# Loop over catalog files (use PKL – most complete)
################################################################################
catalog_files = sorted(glob.glob(
    os.path.join(CATALOG_DIR, "TripleAGN-Catalog-R50-z*.pkl")
))

print(f"Found {len(catalog_files)} catalog files")

for catfile in catalog_files:

    # ---------------- Parse redshift from filename ----------------
    z_match = re.search(r"z([0-9]+\.[0-9]+)", catfile)
    if z_match is None:
        raise RuntimeError(f"Could not parse redshift from {catfile}")
    z = float(z_match.group(1))

    print(f"Processing z = {z:.2f}")

    # ---------------- Load catalog ----------------
    df = pd.read_pickle(catfile)

    # Frequency = number of triple AGN systems
    freq_by_z[z] = len(df)

    if len(df) == 0:
        continue

    # ---------------- Identify tangos timestep ----------------
    # Match closest redshift
    timestep = min(
        sim.timesteps,
        key=lambda ts: abs(ts.redshift - z)
    )

    # Extract snapshot number from extension
    match = re.search(r"\.(\d{6})$", timestep.extension)
    if match is None:
        raise RuntimeError(f"Could not parse snapshot from {timestep.extension}")
    snapshot = match.group(1)

    # ---------------- Loop over triple AGN systems ----------------
    for _, row in df.iterrows():

        # ASSUMPTION (true for your catalogs):
        # Row stores tangos halo index
        tangos_halo_index = row["halo_index"]

        halo = timestep.halos[tangos_halo_index]
        finder_id = halo["finder_id"]

        rows_out.append({
            "snapshot": snapshot,
            "redshift": timestep.redshift,
            "halo_finder_id": finder_id
        })

################################################################################
# Save RTGen target CSV
################################################################################
df_out = pd.DataFrame(rows_out)
df_out.sort_values(["redshift", "halo_finder_id"],
                   ascending=[False, True],
                   inplace=True)

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"Saved RTGen target list to {OUTPUT_CSV}")

################################################################################
# Plot frequency vs redshift
################################################################################
z_vals = sorted(freq_by_z.keys())
freq_vals = [freq_by_z[z] for z in z_vals]

plt.figure(figsize=(7, 5))
plt.plot(z_vals, freq_vals, marker="o")
plt.xlabel("Redshift")
plt.ylabel("Number of Triple AGN Systems")
plt.title("Triple AGN Frequency vs Redshift (Romulus25)")
plt.grid(True)

plt.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved frequency plot to {OUTPUT_PLOT}")
