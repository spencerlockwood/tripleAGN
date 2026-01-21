# extract_romulus_info_for_rtgen.py
"""
Script to extract necessary information from a Romulus25 snapshot
to populate the RTGen input_params.py file
"""

import pynbody as pnb
import numpy as np
import sys

def extract_snapshot_info(snapshot_path, snapshot_name):
    """
    Extract all the information RTGen needs from a Romulus25 snapshot
    """
    print(f"Loading snapshot: {snapshot_path}{snapshot_name}")
    
    # Load snapshot
    sim = pnb.load(f"{snapshot_path}{snapshot_name}")
    
    # Get basic simulation properties
    print("\n" + "="*70)
    print("SIMULATION PROPERTIES")
    print("="*70)
    
    # Redshift and cosmology
    z = sim.properties['z']
    a = sim.properties['a']  # scale factor
    h = sim.properties['h']  # hubble parameter
    omegaM0 = sim.properties['omegaM0']
    omegaL0 = sim.properties['omegaL0']
    
    print(f"Redshift (z): {z}")
    print(f"Scale factor (a): {a}")
    print(f"Hubble parameter (h): {h}")
    print(f"Omega_M: {omegaM0}")
    print(f"Omega_Lambda: {omegaL0}")
    
    # Box size and units
    print(f"\nBox size: {sim.properties['boxsize']}")
    
    # Unit conversions (critical for RTGen)
    print("\n" + "="*70)
    print("UNIT CONVERSIONS")
    print("="*70)
    print(f"Length unit: {sim['pos'].units}")
    print(f"Mass unit: {sim['mass'].units}")
    print(f"Velocity unit: {sim['vel'].units}")
    print(f"Time unit: {sim.properties['time'].units if 'time' in sim.properties else 'N/A'}")
    
    # Particle counts
    print("\n" + "="*70)
    print("PARTICLE COUNTS")
    print("="*70)
    print(f"Gas particles: {len(sim.gas)}")
    print(f"Star particles: {len(sim.star)}")
    print(f"Dark matter particles: {len(sim.dark)}")
    if hasattr(sim, 'bh'):
        print(f"Black hole particles: {len(sim.bh)}")
    
    # Gas properties available
    print("\n" + "="*70)
    print("GAS PARTICLE PROPERTIES")
    print("="*70)
    gas_props = list(sim.gas.keys())
    for prop in sorted(gas_props):
        print(f"  {prop}: {sim.gas[prop].units}")
    
    # Star properties available
    print("\n" + "="*70)
    print("STAR PARTICLE PROPERTIES")
    print("="*70)
    star_props = list(sim.star.keys())
    for prop in sorted(star_props):
        print(f"  {prop}: {sim.star[prop].units}")
    
    # Check for param file
    print("\n" + "="*70)
    print("PARAMETER FILE")
    print("="*70)
    param_file = f"{snapshot_path}{snapshot_name}.param"
    try:
        with open(param_file, 'r') as f:
            print(f"Found param file: {param_file}")
            print("\nContents:")
            print(f.read())
    except FileNotFoundError:
        print(f"No param file found at: {param_file}")
        print("Checking for .AHF_param...")
        try:
            ahf_param = f"{snapshot_path}{snapshot_name}.AHF_param"
            with open(ahf_param, 'r') as f:
                print(f"Found AHF param file: {ahf_param}")
                print("\nContents:")
                print(f.read())
        except FileNotFoundError:
            print("No AHF param file found either")
    
    return {
        'z': z,
        'a': a,
        'h': h,
        'omegaM0': omegaM0,
        'omegaL0': omegaL0,
        'boxsize': sim.properties['boxsize'],
        'n_gas': len(sim.gas),
        'n_star': len(sim.star),
        'n_dark': len(sim.dark),
        'length_unit': str(sim['pos'].units),
        'mass_unit': str(sim['mass'].units),
        'velocity_unit': str(sim['vel'].units),
    }


def extract_triple_agn_region(snapshot_path, snapshot_name, center_pos, region_size=100.0):
    """
    Extract a specific region around a triple AGN system
    
    Parameters:
    -----------
    snapshot_path : str
        Path to snapshot directory
    snapshot_name : str
        Snapshot file name
    center_pos : array-like
        Center position in physical kpc [x, y, z]
    region_size : float
        Size of region to extract in physical kpc
    """
    print(f"\nExtracting region around {center_pos} with size {region_size} kpc")
    
    # Load snapshot
    sim = pnb.load(f"{snapshot_path}{snapshot_name}")
    sim.physical_units()  # Convert to physical units
    
    # Extract spherical region
    region = sim[pnb.filt.Sphere(region_size, center_pos)]
    
    print(f"\nRegion properties:")
    print(f"  Gas particles: {len(region.gas)}")
    print(f"  Star particles: {len(region.star)}")
    print(f"  Total gas mass: {region.gas['mass'].sum()}")
    print(f"  Total stellar mass: {region.star['mass'].sum()}")
    
    # Gas properties statistics
    if len(region.gas) > 0:
        print(f"\n  Gas density range: {region.gas['rho'].min()} to {region.gas['rho'].max()}")
        print(f"  Gas temperature range: {region.gas['temp'].min()} to {region.gas['temp'].max()} K")
        if 'metals' in region.gas.keys():
            print(f"  Gas metallicity range: {region.gas['metals'].min()} to {region.gas['metals'].max()}")
    
    # Star properties statistics
    if len(region.star) > 0:
        print(f"\n  Stellar age range: {region.star['age'].min()} to {region.star['age'].max()} Gyr")
        if 'metals' in region.star.keys():
            print(f"  Stellar metallicity range: {region.star['metals'].min()} to {region.star['metals'].max()}")
    
    return region


def create_rtgen_input_template(snapshot_info, triple_agn_data, output_file="input_params_tripleAGN.py"):
    """
    Create a template input_params.py file for RTGen based on extracted info
    """
    
    with open(output_file, 'w') as f:
        f.write("""# RTGen input parameters for Romulus25 Triple AGN System
# Auto-generated from extract_romulus_info_for_rtgen.py

import numpy as np

# ============================================================================
# SNAPSHOT INFORMATION
# ============================================================================
""")
        f.write(f"snapshot_path = '{triple_agn_data['snapshot_path']}'\n")
        f.write(f"snapshot_name = '{triple_agn_data['snapshot_name']}'\n")
        f.write(f"redshift = {snapshot_info['z']:.4f}\n")
        f.write(f"scale_factor = {snapshot_info['a']:.6f}\n")
        f.write(f"hubble_param = {snapshot_info['h']:.4f}\n\n")
        
        f.write("""# ============================================================================
# TRIPLE AGN SYSTEM
# ============================================================================
""")
        f.write(f"# System center (physical kpc)\n")
        f.write(f"system_center = np.array({list(triple_agn_data['center'])})\n\n")
        
        f.write(f"# Black hole positions (physical kpc)\n")
        f.write(f"BH1_pos = np.array({list(triple_agn_data['BH1_pos'])})\n")
        f.write(f"BH2_pos = np.array({list(triple_agn_data['BH2_pos'])})\n")
        f.write(f"BH3_pos = np.array({list(triple_agn_data['BH3_pos'])})\n\n")
        
        f.write(f"# Black hole properties\n")
        f.write(f"BH1_mass = {triple_agn_data['BH1_mass']:.2e}  # Msun\n")
        f.write(f"BH2_mass = {triple_agn_data['BH2_mass']:.2e}  # Msun\n")
        f.write(f"BH3_mass = {triple_agn_data['BH3_mass']:.2e}  # Msun\n\n")
        
        f.write(f"BH1_Lbol = {triple_agn_data['BH1_Lbol']:.2e}  # erg/s\n")
        f.write(f"BH2_Lbol = {triple_agn_data['BH2_Lbol']:.2e}  # erg/s\n")
        f.write(f"BH3_Lbol = {triple_agn_data['BH3_Lbol']:.2e}  # erg/s\n\n")
        
        f.write("""# ============================================================================
# REGION TO EXTRACT
# ============================================================================
region_size = 100.0  # kpc (physical) - adjust based on system size
grid_resolution = 128  # cells per dimension - start coarse, increase later

# ============================================================================
# DUST PARAMETERS (adjust as needed)
# ============================================================================
dust_to_gas_ratio = 0.01  # Standard MW value

# ============================================================================
# RADIATIVE TRANSFER PARAMETERS (adjust as needed)
# ============================================================================
n_photons = 1e6  # Number of photon packages
lambda_min = 0.1  # microns
lambda_max = 1000.0  # microns

# ============================================================================
# OBSERVATION PARAMETERS (adjust as needed)
# ============================================================================
inclination = 45.0  # degrees
position_angle = 0.0  # degrees
image_npix = 512
image_size_kpc = 150.0
""")
    
    print(f"\nTemplate input file created: {output_file}")


if __name__ == "__main__":
    # Example usage
    snapshot_path = "/home/stlock/projects/rrg-babul-ad/SHARED/Romulus/cosmo25/"
    
    if len(sys.argv) > 1:
        snapshot_name = sys.argv[1]
    else:
        snapshot_name = "cosmo25p.768sg1bwK1BHe75.000824"  # Example
    
    # Extract general snapshot info
    info = extract_snapshot_info(snapshot_path, snapshot_name)
    
    print("\n" + "="*70)
    print("NEXT STEPS FOR RTGEN")
    print("="*70)
    print("""
1. The snapshot format is Tipsy (ChaNGa output) - confirmed compatible with RTGen

2. You need to provide RTGen with:
   - Snapshot file path
   - Center position of your triple AGN system (from your catalog)
   - BH positions and properties (from your catalog)
   - Region size to extract
   - Grid resolution for RADMC-3D

3. RTGen will need to read:
   - Gas particle positions, masses, densities, temperatures, metallicities
   - Star particle positions, masses, ages, metallicities
   - BH positions (to add as point sources)

4. For each triple AGN system from your catalog, create an input_params.py
   with the specific system information.

5. Run the RTGen pipeline following their workflow:
   a) SPH interpolation (reads Tipsy, creates RADMC-3D grid)
   b) Stellar modeling (creates stellar sources for RADMC-3D)
   c) Run RADMC-3D radiative transfer
   d) Post-processing (create mock observations)
""")