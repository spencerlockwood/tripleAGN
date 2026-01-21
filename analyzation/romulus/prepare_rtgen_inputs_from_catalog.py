# prepare_rtgen_inputs_from_catalog.py
"""
Prepare RTGen input files from triple AGN catalog
"""

import pandas as pd
import numpy as np
import os

def prepare_system_for_rtgen(system_row, output_dir="/scratch/stlock/tripleAGNs/RTGen_inputs/"):
    """
    Prepare input parameters for a single triple AGN system
    """
    
    # Extract system info
    z = system_row['redshift']
    snapshot = system_row['snapshot']
    snapshot_fullname = system_row['snapshot_fullname']
    
    # Scale factor for comoving -> physical conversion
    a = 1.0 / (1.0 + z)
    
    # BH positions (convert from comoving to physical)
    BH1_pos = np.array([
        system_row['BH_x'] * a,
        system_row['BH_y'] * a,
        system_row['BH_z'] * a
    ])
    
    BH2_pos = np.array([
        system_row['Neighbour1BH_x'] * a,
        system_row['Neighbour1BH_y'] * a,
        system_row['Neighbour1BH_z'] * a
    ])
    
    BH3_pos = np.array([
        system_row['Neighbour2BH_x'] * a,
        system_row['Neighbour2BH_y'] * a,
        system_row['Neighbour2BH_z'] * a
    ])
    
    # System center (centroid of the three BHs)
    center = (BH1_pos + BH2_pos + BH3_pos) / 3.0
    
    # BH properties
    BH1_mass = system_row['BH1_mass']
    BH2_mass = system_row['BH2_mass']
    BH3_mass = system_row['BH3_mass']
    
    BH1_Lbol = system_row['BH1_Lbol']
    BH2_Lbol = system_row['BH2_Lbol']
    BH3_Lbol = system_row['BH3_Lbol']
    
    # BH IDs for tracking
    BH1_id = int(system_row['BH1_finder_id'])
    BH2_id = int(system_row['BH2_finder_id'])
    BH3_id = int(system_row['BH3_finder_id'])
    
    # Determine region size (5x the maximum separation)
    max_sep_phys = system_row['max_separation'] * a  # Convert to physical
    region_size = max(100.0, 5.0 * max_sep_phys)  # At least 100 kpc
    
    # Create output directory
    system_dir = os.path.join(output_dir, f"z{z:.2f}_snap{snapshot}")
    os.makedirs(system_dir, exist_ok=True)
    
    # Write input_params.py for this system
    input_file = os.path.join(system_dir, "input_params.py")
    
    with open(input_file, 'w') as f:
        f.write(f"""# RTGen input parameters for Triple AGN System
# Redshift: {z:.4f}
# Snapshot: {snapshot_fullname}
# Generated from triple AGN catalog

import numpy as np

# ============================================================================
# SNAPSHOT INFORMATION
# ============================================================================
snapshot_path = "/home/stlock/projects/rrg-babul-ad/SHARED/Romulus/cosmo25/"
snapshot_name = "{snapshot_fullname}"
redshift = {z:.4f}
scale_factor = {a:.6f}

# ============================================================================
# TRIPLE AGN SYSTEM
# ============================================================================
# System ID
system_index = {system_row['system_index']}

# System center (physical kpc)
system_center = np.array({list(center)})

# Black hole particle IDs (pynbody finder_id)
BH1_id = {BH1_id}
BH2_id = {BH2_id}
BH3_id = {BH3_id}

# Black hole positions (physical kpc)
BH1_pos = np.array({list(BH1_pos)})
BH2_pos = np.array({list(BH2_pos)})
BH3_pos = np.array({list(BH3_pos)})

# Black hole masses (Msun)
BH1_mass = {BH1_mass:.6e}
BH2_mass = {BH2_mass:.6e}
BH3_mass = {BH3_mass:.6e}

# Black hole bolometric luminosities (erg/s)
BH1_Lbol = {BH1_Lbol:.6e}
BH2_Lbol = {BH2_Lbol:.6e}
BH3_Lbol = {BH3_Lbol:.6e}

# Separations (physical kpc)
sep_BH1_BH2 = {system_row['sep_BH1_BH2'] * a:.2f}
sep_BH1_BH3 = {system_row['sep_BH1_BH3'] * a:.2f}
sep_BH2_BH3 = {system_row['sep_BH2_BH3'] * a:.2f}

# ============================================================================
# REGION TO EXTRACT
# ============================================================================
region_size = {region_size:.1f}  # kpc (physical)
grid_resolution = 128  # cells per dimension - start coarse

# Grid boundaries (physical kpc, centered on system_center)
x_min, x_max = -region_size/2, region_size/2
y_min, y_max = -region_size/2, region_size/2
z_min, z_max = -region_size/2, region_size/2

# ============================================================================
# DUST PARAMETERS
# ============================================================================
dust_to_gas_ratio = 0.01  # Standard MW value
dust_composition = ['silicate', 'graphite']
dust_fractions = [0.75, 0.25]

# ============================================================================
# STELLAR PARAMETERS
# ============================================================================
use_stellar_particles = True
stellar_library = 'PEGASE'  # or 'STARBURST99'

# ============================================================================
# AGN PARAMETERS
# ============================================================================
use_AGN_sources = True
AGN_SED_type = 'power_law'
AGN_spectral_index = -1.5

# ============================================================================
# RADIATIVE TRANSFER PARAMETERS
# ============================================================================
lambda_min = 0.1  # microns (UV)
lambda_max = 1000.0  # microns (sub-mm)
n_wavelengths = 100

n_photons_thermal = 1e6
n_photons_scat = 1e5
n_photons_spec = 1e6

# ============================================================================
# OBSERVATION PARAMETERS
# ============================================================================
inclination = 45.0  # degrees from face-on
position_angle = 0.0  # degrees

image_npix = 512
image_size_kpc = {region_size * 1.5:.1f}  # slightly larger than region

# Simulated instruments
simulate_JWST = True
JWST_filters = ['F150W', 'F277W', 'F444W']

simulate_ALMA = False  # Set to True if doing line transfer

# ============================================================================
# OUTPUT
# ============================================================================
output_dir = "/scratch/stlock/tripleAGNs/RTGen_output/z{z:.2f}_snap{snapshot}/"
""")
    
    print(f"Created input file: {input_file}")
    
    return {
        'system_dir': system_dir,
        'input_file': input_file,
        'snapshot': snapshot_fullname,
        'center': center,
        'region_size': region_size
    }


def main():
    # Load triple AGN catalog
    catalog_file = "/scratch/stlock/tripleAGNs/plots_and_data/triple_agn_system_ids.csv"
    systems = pd.read_csv(catalog_file)
    
    print(f"Loaded {len(systems)} triple AGN systems from catalog")
    
    # Select systems to process
    # Start with low-redshift systems (easier to resolve)
    selected = systems[systems['redshift'] < 1.0].copy()
    
    print(f"\nPreparing RTGen inputs for {len(selected)} systems (z < 1.0)")
    
    # Prepare each system
    prepared_systems = []
    for idx, row in selected.iterrows():
        print(f"\nSystem {idx + 1}/{len(selected)}: z={row['redshift']:.2f}")
        result = prepare_system_for_rtgen(row)
        prepared_systems.append(result)
    
    print(f"\n{'='*70}")
    print(f"Prepared {len(prepared_systems)} systems for RTGen")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("1. Copy one of the input_params.py files to RTGen-public/input/")
    print("2. Run the RTGen pipeline following their README")
    print("3. Check output images and SEDs")


if __name__ == "__main__":
    main()