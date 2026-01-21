"""
import pandas as pd

df = pd.read_pickle("/scratch/stlock/tripleAGNs/catalogs/1e41lum/ID_correct_catalogue_15kpc_1e41lum/TripleAGN-Catalog-R50-z2.93.pkl")

row = df.iloc[0]

snapshot = row['snapshot']
halo_id = int(row['pynbody_haloid'])

print(snapshot, halo_id)
"""


import pandas as pd
import pynbody
from pynbody.halo.ahf import AHFCatalogue

# ---- USER INPUTS ----
catalog_path = "/scratch/stlock/tripleAGNs/catalogs/1e41lum/ID_correct_catalogue_15kpc_1e41lum/TripleAGN-Catalog-R50-z2.93.pkl"
snapshot_base = "/home/stlock/projects/rrg-babul-ad/SHARED/Romulus/cosmo25/"
# ---------------------

# Load catalog
df = pd.read_pickle(catalog_path)

# Pick ONE system to test
row = df.iloc[0]

snapshot = row['snapshot']               # e.g. '000105'
#snapshot = "001302"
halo_id = int(row['pynbody_haloid'])      # AHF halo ID
#halo_id = 76443

print(f"Testing snapshot {snapshot}, halo {halo_id}")

# Load snapshot
snap_path = f"{snapshot_base}/cosmo25p.768sg1bwK1BHe75.{snapshot}"
s = pynbody.load(snap_path)
s.physical_units()

# Load AHF halos explicitly
h = AHFCatalogue(s, ignore_missing_substructure=True)

print("Number of halos:", len(h))
print("Target halo object:")
print(h[halo_id])
