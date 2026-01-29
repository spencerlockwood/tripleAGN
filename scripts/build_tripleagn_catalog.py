"""
build_tripleagn_catalog.py
==========================

This script builds a catalog of triple AGN systems from the Romulus25 simulation.

TRIPLE AGN DEFINITION:
- A HOST AGN (Lbol > 10^43 erg/s)
- With EXACTLY TWO neighboring AGN (each also Lbol > 10^43 erg/s) within 
  the separation threshold (default 30 kpc) of the host
- The two neighbors do NOT need to be within the threshold of each other,
  only within the threshold of the host
- No AGN can be part of more than one triple system

HANDLING COMPLEX CASES:
- If an AGN could be a host for multiple triples, it's assigned to one triple
- If an AGN is already assigned as a neighbor in one triple, it cannot be 
  a host or neighbor in another triple
- Priority is given to systems where the host has the highest luminosity

Usage:
    python build_tripleagn_catalog.py <snapshot_index>

For paper comparison, use SLURM --array=62,65,69,75,80,88,96,107,119
which corresponds to z = 2.0, 1.79, 1.5, 1.22, 1.0, 0.75, 0.5, 0.25, 0.05
"""

print('start running the script')
import sys
sys.path.insert(0, "/home/stlock/XIGrM")
import numpy as np
import pandas as pd
import pynbody as pnb
import tangos
import h5py
import os
import XIGrM.cosmology as cos
import XIGrM.calculate_R as cR
import argparse
import scipy.constants as const
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict


##############################################
# STEP 1: Load snapshot and redshift info
##############################################
def load_snapshot(snapshot_index, snapshot_list, base_path):
    print('step 1: loading snapshot')
    snap = snapshot_list[snapshot_index]
    file_path = os.path.join(base_path, f"cosmo25p.768sg1bwK1BHe75.{snap}")
    s = pnb.load(file_path)
    s.physical_units()
    z = s.properties['z']
    age_gyr = float(pnb.analysis.cosmology.age(s, z=None, unit='Gyr'))
    return s, snap, z, age_gyr


##############################################
# STEP 2: Load Tangos BH data from HDF5
##############################################
def load_tangos_bh_data(snap, hdf5_path):
    print('step 2: loading tangos BH data')
    with h5py.File(hdf5_path.format(snap), 'r') as hf:
        data = {k: np.array(hf[k]) for k in hf.keys()}
    return pd.DataFrame(data)


##############################################
# STEP 3: Match BHs between Tangos and Pynbody
##############################################
def match_bh_properties(sim, tangos_df):
    print('step 3: matching BH properties')
    BH_idx = np.where(sim.s['tform'] < 0)
    all_bh_ids = sim.s['iord'][BH_idx]
    match_mask = np.isin(all_bh_ids, tangos_df['BH_id'].values)

    matched_ids = sim.s['iord'][BH_idx][match_mask]
    matched_vel = sim.s['vel'][BH_idx][match_mask]
    matched_pos = sim.s['pos'][BH_idx][match_mask].in_units('kpc')

    matched_df = pd.DataFrame({
        'BH_id': matched_ids,
        'BH_vx': matched_vel[:, 0],
        'BH_vy': matched_vel[:, 1],
        'BH_vz': matched_vel[:, 2],
        'BH_x': np.array(matched_pos[:, 0]),
        'BH_y': np.array(matched_pos[:, 1]),
        'BH_z': np.array(matched_pos[:, 2]),
    })

    merged = pd.merge(tangos_df, matched_df, on='BH_id', how='inner')
    return merged


##############################################
# STEP 4: Compute BH local environment properties
##############################################
def compute_bh_local_env(sim, pos, t_now, t_prev):
    """Compute local environment properties around a BH position."""
    result = {}
    for r in [1, 5]:
        region = sim[pnb.filt.Sphere(r, pos)]
        mstar = region.s['mass'].sum()
        mgas = region.gas['mass'].sum()
        volume = (4/3) * np.pi * (r**3)

        result[f'BH_Mstar_{r}'] = mstar
        result[f'BH_GasDensity_{r}'] = mgas / volume if volume > 0 else np.nan

        if len(region.gas) > 0:
            gas_mass = region.gas['mass']
            gas_metal = region.gas['metals'] * gas_mass
            result[f'BH_GasMetallicity_{r}'] = gas_metal.sum() / gas_mass.sum() if gas_mass.sum() > 0 else np.nan
        else:
            result[f'BH_GasMetallicity_{r}'] = np.nan

        if len(region.s) > 0:
            star_mass = region.s['mass']
            star_metal = region.s['metals'] * star_mass
            result[f'BH_StellarMetallicity_{r}'] = star_metal.sum() / star_mass.sum() if star_mass.sum() > 0 else np.nan
        else:
            result[f'BH_StellarMetallicity_{r}'] = np.nan

        newstars = region.s['tform'] > t_prev
        result[f'BH_SFR_{r}'] = region.s['mass'][newstars].sum() / ((t_now - t_prev) * 1e9) if len(region.s['mass'][newstars]) > 0 else 0

    return result


##############################################
# STEP 5: Identify triple AGN systems
##############################################
def find_triple_agn_systems(df, sim, t_now, t_prev, separation_threshold=30, lbol_threshold=1e43):
    """
    Identify triple AGN systems based on the specified definition.
    
    TRIPLE AGN DEFINITION:
    - A HOST AGN with EXACTLY TWO neighboring AGN within separation_threshold
    - Neighbors must be AGN (above lbol_threshold) but do NOT need to be 
      within separation_threshold of each other
    - No AGN can be part of more than one triple system
    
    ALGORITHM:
    1. Find all AGN above luminosity threshold
    2. For each AGN, count how many other AGN are within separation_threshold
    3. Identify potential hosts (AGN with exactly 2 neighbors within threshold)
    4. Sort potential hosts by luminosity (brightest first)
    5. Greedily assign triples, marking AGN as "used" once assigned
    
    Parameters:
    -----------
    df : DataFrame
        Merged BH dataframe with positions and properties
    sim : pynbody SimSnap
        The simulation snapshot for environment calculations
    t_now : float
        Current cosmic time in Gyr
    t_prev : float
        Previous snapshot cosmic time in Gyr
    separation_threshold : float
        Maximum separation in kpc for neighbors from host (default: 30)
    lbol_threshold : float
        Minimum bolometric luminosity in erg/s to classify as AGN (default: 1e43)
    
    Returns:
    --------
    DataFrame with triple AGN systems
    """
    print('step 5: identifying triple AGN systems')

    df = df.copy()

    # Physical constants for luminosity calculation
    msun_to_g = 1.989e33
    yr_to_s = 3.154e7
    c_cgs = 3e10

    # Compute bolometric luminosity and Eddington luminosity
    df['BH_Lbol'] = 0.1 * df['BH_mdot'] * msun_to_g / yr_to_s * c_cgs**2
    df['BH_Ledd'] = 1.26e38 * df['BH_mass']

    # -----------------------------------------------------------------
    # STEP 5a: Select AGN by luminosity threshold
    # -----------------------------------------------------------------
    agn_mask = df['BH_Lbol'] > lbol_threshold
    agn_df = df[agn_mask].copy().reset_index(drop=True)
    
    n_agn = len(agn_df)
    print(f"  Total AGN (Lbol > {lbol_threshold:.2e}): {n_agn}")

    if n_agn < 3:
        print("  Not enough AGN to form triples (need at least 3). Returning empty DataFrame.")
        return pd.DataFrame()

    # -----------------------------------------------------------------
    # STEP 5b: Compute pairwise distances between all AGN
    # -----------------------------------------------------------------
    pos_array = agn_df[['BH_x', 'BH_y', 'BH_z']].values
    ids = agn_df['BH_id'].values
    lbols = agn_df['BH_Lbol'].values
    
    # Create ID to index mapping
    id_to_idx = {bhid: idx for idx, bhid in enumerate(ids)}
    
    # Compute full distance matrix
    dist_matrix = squareform(pdist(pos_array))

    # -----------------------------------------------------------------
    # STEP 5c: For each AGN, find neighbors within separation threshold
    # -----------------------------------------------------------------
    neighbors_dict = {}  # {host_id: [list of neighbor ids within threshold]}
    
    for i, host_id in enumerate(ids):
        neighbors = []
        for j, neighbor_id in enumerate(ids):
            if i != j and dist_matrix[i, j] < separation_threshold:
                neighbors.append(neighbor_id)
        neighbors_dict[host_id] = neighbors

    # -----------------------------------------------------------------
    # STEP 5d: Identify potential triple hosts (AGN with exactly 2 neighbors)
    # -----------------------------------------------------------------
    potential_hosts = []
    
    for host_id, neighbors in neighbors_dict.items():
        if len(neighbors) == 2:
            host_idx = id_to_idx[host_id]
            host_lbol = lbols[host_idx]
            potential_hosts.append({
                'host_id': host_id,
                'host_lbol': host_lbol,
                'neighbor_ids': neighbors
            })
    
    print(f"  Potential triple hosts (AGN with exactly 2 neighbors): {len(potential_hosts)}")
    
    if len(potential_hosts) == 0:
        print("  No potential triple hosts found. Returning empty DataFrame.")
        return pd.DataFrame()

    # -----------------------------------------------------------------
    # STEP 5e: Sort potential hosts by luminosity (brightest first)
    # This gives priority to brighter hosts when there's competition
    # -----------------------------------------------------------------
    potential_hosts.sort(key=lambda x: x['host_lbol'], reverse=True)

    # -----------------------------------------------------------------
    # STEP 5f: Greedily assign triples, ensuring no AGN is used twice
    # -----------------------------------------------------------------
    used_agn = set()  # Track which AGN have been assigned to a triple
    confirmed_triples = []
    
    for candidate in potential_hosts:
        host_id = candidate['host_id']
        neighbor_ids = candidate['neighbor_ids']
        
        # Check if host or any neighbor is already used
        if host_id in used_agn:
            continue
        if any(nid in used_agn for nid in neighbor_ids):
            continue
        
        # This is a valid triple - assign it
        confirmed_triples.append({
            'host_id': host_id,
            'neighbor1_id': neighbor_ids[0],
            'neighbor2_id': neighbor_ids[1]
        })
        
        # Mark all three AGN as used
        used_agn.add(host_id)
        used_agn.add(neighbor_ids[0])
        used_agn.add(neighbor_ids[1])
    
    print(f"  Confirmed triple AGN systems: {len(confirmed_triples)}")
    
    if len(confirmed_triples) == 0:
        print("  No triple systems confirmed after deduplication. Returning empty DataFrame.")
        return pd.DataFrame()

    # -----------------------------------------------------------------
    # STEP 5g: Build output DataFrame with full properties
    # -----------------------------------------------------------------
    triples_output = []
    
    for triple_idx, triple in enumerate(confirmed_triples):
        host_id = triple['host_id']
        n1_id = triple['neighbor1_id']
        n2_id = triple['neighbor2_id']
        
        # Get data for all three AGN
        host_data = agn_df[agn_df['BH_id'] == host_id].iloc[0]
        n1_data = agn_df[agn_df['BH_id'] == n1_id].iloc[0]
        n2_data = agn_df[agn_df['BH_id'] == n2_id].iloc[0]
        
        # Sort neighbors by luminosity (brighter = Neighbour1)
        if n1_data['BH_Lbol'] < n2_data['BH_Lbol']:
            n1_data, n2_data = n2_data, n1_data
            n1_id, n2_id = n2_id, n1_id
        
        # Compute separations
        # Host to Neighbour1
        dx1 = host_data['BH_x'] - n1_data['BH_x']
        dy1 = host_data['BH_y'] - n1_data['BH_y']
        dz1 = host_data['BH_z'] - n1_data['BH_z']
        sep_host_n1_3d = np.sqrt(dx1**2 + dy1**2 + dz1**2)
        sep_host_n1_2d = np.sqrt(dx1**2 + dy1**2)
        
        # Host to Neighbour2
        dx2 = host_data['BH_x'] - n2_data['BH_x']
        dy2 = host_data['BH_y'] - n2_data['BH_y']
        dz2 = host_data['BH_z'] - n2_data['BH_z']
        sep_host_n2_3d = np.sqrt(dx2**2 + dy2**2 + dz2**2)
        sep_host_n2_2d = np.sqrt(dx2**2 + dy2**2)
        
        # Neighbour1 to Neighbour2
        dx12 = n1_data['BH_x'] - n2_data['BH_x']
        dy12 = n1_data['BH_y'] - n2_data['BH_y']
        dz12 = n1_data['BH_z'] - n2_data['BH_z']
        sep_n1_n2_3d = np.sqrt(dx12**2 + dy12**2 + dz12**2)
        sep_n1_n2_2d = np.sqrt(dx12**2 + dy12**2)
        
        # Compute local environment for host AGN
        print(f'  Computing environment for triple {triple_idx + 1}: host BH_id={int(host_id)}')
        host_pos = [host_data['BH_x'], host_data['BH_y'], host_data['BH_z']]
        host_env = compute_bh_local_env(sim, host_pos, t_now, t_prev)
        
        # Compute local environment for Neighbour1
        n1_pos = [n1_data['BH_x'], n1_data['BH_y'], n1_data['BH_z']]
        n1_env = compute_bh_local_env(sim, n1_pos, t_now, t_prev)
        n1_env = {f'Neighbour1{key}': val for key, val in n1_env.items()}
        
        # Compute local environment for Neighbour2
        n2_pos = [n2_data['BH_x'], n2_data['BH_y'], n2_data['BH_z']]
        n2_env = compute_bh_local_env(sim, n2_pos, t_now, t_prev)
        n2_env = {f'Neighbour2{key}': val for key, val in n2_env.items()}

        # Build the triple record
        # Host AGN properties (primary)
        triple_record = host_data.to_dict()
        triple_record.update(host_env)
        
        # Neighbour1 properties
        n1_dict = {f'Neighbour1{col}': n1_data[col] 
                   for col in df.columns 
                   if col.startswith('BH_') or col == 'BH_id'}
        n1_dict.update({
            'Separation_Host_N1_3D_kpc': sep_host_n1_3d,
            'Separation_Host_N1_2D_kpc': sep_host_n1_2d,
            'Neighbour1BH_pynbody_haloid': n1_data.get('pynbody_haloid', np.nan)
        })
        triple_record.update(n1_dict)
        triple_record.update(n1_env)
        
        # Neighbour2 properties
        n2_dict = {f'Neighbour2{col}': n2_data[col] 
                   for col in df.columns 
                   if col.startswith('BH_') or col == 'BH_id'}
        n2_dict.update({
            'Separation_Host_N2_3D_kpc': sep_host_n2_3d,
            'Separation_Host_N2_2D_kpc': sep_host_n2_2d,
            'Neighbour2BH_pynbody_haloid': n2_data.get('pynbody_haloid', np.nan)
        })
        triple_record.update(n2_dict)
        triple_record.update(n2_env)
        
        # Inter-neighbor separation
        triple_record['Separation_N1_N2_3D_kpc'] = sep_n1_n2_3d
        triple_record['Separation_N1_N2_2D_kpc'] = sep_n1_n2_2d
        
        triples_output.append(triple_record)

    print(f"  Total triple AGN systems found: {len(triples_output)}")

    return pd.DataFrame(triples_output)


##############################################
# STEP 6: Compute halo properties
##############################################
def compute_halo_environment_properties(sim, halo_ids, t_now, t_prev):
    print('step 6: computing halo environment properties')
    h = sim.halos(ahf_mpi=True)
    results = []

    # Sanitize halo_ids: drop NaN, ensure integers and unique
    try:
        halo_ids_arr = np.asarray(halo_ids)
    except Exception:
        halo_ids_arr = np.array(list(halo_ids))

    # Drop NaNs and convert to ints
    halo_ids_clean = []
    for x in np.unique(halo_ids_arr):
        if pd.isna(x):
            continue
        try:
            halo_ids_clean.append(int(x))
        except Exception:
            continue

    print(f"  Processing {len(halo_ids_clean)} unique halos")

    for haloid in halo_ids_clean:
        try:
            center = pnb.analysis.halo.center(h[haloid], mode='pot', retcen=True).in_units('kpc')
            virovdens = cos.Delta_vir(sim)
            rdict = {'vir': virovdens, '200': 200, '500': 500, '2500': 2500}
            MassRadii = cR.get_radius(h[haloid], list(rdict.values()), prop=sim.properties, cen=center)

            def compute_sfr(sub):
                newstars = sub.s['tform'] > t_prev
                return sub.s['mass'][newstars].sum() / ((t_now - t_prev) * 1e9)

            sub25 = h[haloid][pnb.filt.Sphere(25, center)]
            sub30 = h[haloid][pnb.filt.Sphere(30, center)]
            sub50 = h[haloid][pnb.filt.Sphere(50, center)]

            def half_mass_radius():
                radii = np.linspace(0.1, 0.3 * MassRadii[1][200.0], 100)
                masses = []
                for r in radii:
                    mass = h[haloid][pnb.filt.Sphere(r, center)].s['mass'].sum()
                    masses.append(mass)
                masses = np.array(masses)
                total_mass = masses[-1]
                half_mass = total_mass / 2
                idx = np.where(masses >= half_mass)[0][0]
                return 4.0 * radii[idx]

            Rgal = half_mass_radius()
            subgal = h[haloid][pnb.filt.Sphere(Rgal, center)]

            results.append({
                'Halo_id': int(haloid),
                'Halo_center_x': center[0],
                'Halo_center_y': center[1],
                'Halo_center_z': center[2],
                'Halo_Mvir': MassRadii[0][virovdens],
                'Halo_Rvir': MassRadii[1][virovdens],
                'Halo_M200': MassRadii[0][200.0],
                'Halo_R200': MassRadii[1][200.0],
                'Halo_M500': MassRadii[0][500.0],
                'Halo_R500': MassRadii[1][500.0],
                'Halo_M2500': MassRadii[0][2500.0],
                'Halo_R2500': MassRadii[1][2500.0],
                'Halo_Mstar25': sub25.s['mass'].sum(),
                'Halo_GasDensity25': sub25.g['mass'].sum(),
                'Halo_SFR25': compute_sfr(sub25),
                'Halo_Mstar30': sub30.s['mass'].sum(),
                'Halo_GasDensity30': sub30.g['mass'].sum(),
                'Halo_SFR30': compute_sfr(sub30),
                'Halo_Mstar50': sub50.s['mass'].sum(),
                'Halo_GasDensity50': sub50.g['mass'].sum(),
                'Halo_SFR50': compute_sfr(sub50),
                'Halo_Mgal': subgal['mass'].sum(),
                'Halo_MgasRgal': subgal.g['mass'].sum(),
                'Halo_MstarRgal': subgal.s['mass'].sum(),
                'Halo_SFRRgal': compute_sfr(subgal),
            })

        except Exception as e:
            print(f"  Failed halo {haloid}: {e}")

    df = pd.DataFrame(results)

    if df.empty:
        return pd.DataFrame(columns=['Halo_id']).set_index('Halo_id')
    else:
        return df.drop_duplicates('Halo_id').set_index('Halo_id')

        
##############################################
# STEP 7: Merge AGN catalog with halo properties
##############################################
def merge_agn_with_halos(agn_df, halo_df):
    print('step 7: merging AGN catalog with halo properties')
    
    if agn_df.empty:
        print("  No AGN triples to merge. Returning empty DataFrame.")
        return agn_df
    
    # Merge for host halo
    df = pd.merge(agn_df, halo_df, how='left', left_on='pynbody_haloid', right_index=True)
    
    # Merge for Neighbour1 halo
    df = pd.merge(df, halo_df.add_prefix('Neighbour1Halo_'), how='left', 
                  left_on='Neighbour1BH_pynbody_haloid', right_index=True)
    
    # Merge for Neighbour2 halo
    df = pd.merge(df, halo_df.add_prefix('Neighbour2Halo_'), how='left', 
                  left_on='Neighbour2BH_pynbody_haloid', right_index=True)
    
    return df


##############################################
# STEP 8: Save final catalog
##############################################
def save_triple_agn_catalog(df, output_path_template, z):
    print('step 8: saving catalog')

    # Format file paths
    base = output_path_template.format(z)
    csv_path = base.replace(".pkl", ".csv")
    h5_path = base.replace(".pkl", ".h5")
    pkl_path = base

    # Save files
    df.to_pickle(pkl_path)
    df.to_csv(csv_path, index=False)
    df.to_hdf(h5_path, key="tripleagn", mode="w")

    print(f"Saved catalog for redshift {z:.2f} to:")
    print(f"  - Pickle: {pkl_path}")
    print(f"  - CSV:    {csv_path}")
    print(f"  - HDF5:   {h5_path}")
    print(f"  - Total triple AGN systems: {len(df)}")


##############################################
# MAIN EXECUTION FLOW
##############################################
def main(snapshot_index):
    
    snapshot_list = [
        '000098','000105','000111','000118','000126','000134','000142','000151',
        '000160','000170','000181','000192','000204','000216','000229','000243',
        '000256','000258','000274','000290','000308','000372','000437','000446',
        '000463','000491','000512','000520','000547','000551','000584','000618',
        '000655','000690','000694','000735','000768','000778','000824','000873',
        '000909','000924','000979','001024','001036','001065','001097','001162',
        '001230','001270','001280','001302','001378','001458','001536','001543',
        '001550','001632','001726','001792','001826','001931','001945','002042',
        '002048','002159','002281','002304','002411','002536','002547','002560',
        '002690','002816','002840','002998','003072','003163','003328','003336',
        '003478','003517','003584','003707','003840','003905','004096','004111',
        '004173','004326','004352','004549','004608','004781','004864','005022',
        '005107','005120','005271','005376','005529','005632','005795','005888',
        '006069','006144','006350','006390','006400','006640','006656','006912',
        '006937','007168','007212','007241','007394','007424','007552','007680',
        '007779','007869','007936','008192'
    ]

    # File paths - NOTE: Changed output path for triple AGN
    hdf5_path = "/scratch/stlock/halomap_files/HaloBH-TangosPynbodyMap-R25-snap{}.hdf5"
    output_path = "/scratch/stlock/tripleAGNs/catalogs/1e43lum/catalogue_30kpc_all/TripleAGN-Catalog-R25-z{:.2f}.pkl"
    sim_path = "/home/stlock/projects/rrg-babul-ad/SHARED/Romulus/cosmo25/"

    print("="*60)
    print(f"Processing snapshot index: {snapshot_index}")
    print(f"Snapshot: {snapshot_list[snapshot_index]}")
    print("="*60)

    # Load current and previous snapshots
    print('\nLoading snapshots...')
    s, snap, z, t_now = load_snapshot(snapshot_index, snapshot_list, sim_path)
    print(f"  Current snapshot: {snap}, z = {z:.4f}, t = {t_now:.4f} Gyr")
    
    s_last, snap_prev, z_prev, t_prev = load_snapshot(snapshot_index - 1, snapshot_list, sim_path)
    print(f"  Previous snapshot: {snap_prev}, z = {z_prev:.4f}, t = {t_prev:.4f} Gyr")

    # Load Tangos BH data
    print('\nLoading Tangos BH data...')
    tangos_df = load_tangos_bh_data(snap, hdf5_path)
    print(f"  Loaded {len(tangos_df)} BHs from Tangos")

    # Match BH properties between Tangos and Pynbody
    print('\nMatching BH properties...')
    matched_bh_df = match_bh_properties(s, tangos_df)
    print(f"  Matched {len(matched_bh_df)} BHs")

    # Find triple AGN systems
    print('\nFinding triple AGN systems...')
    agn_triples_df = find_triple_agn_systems(matched_bh_df, s, t_now, t_prev)

    # Handle case of no triple AGN found
    if agn_triples_df.empty:
        print("\nNo triple AGN systems found in this snapshot.")
        print("Saving empty catalog...")
        save_triple_agn_catalog(agn_triples_df, output_path, z)
        return

    # Compute halo environment properties
    print('\nComputing halo properties...')
    all_halo_ids = pd.unique(
        agn_triples_df[['pynbody_haloid', 'Neighbour1BH_pynbody_haloid', 'Neighbour2BH_pynbody_haloid']].values.ravel()
    )
    halo_df = compute_halo_environment_properties(s, all_halo_ids, t_now, t_prev)

    # Merge AGN catalog with halo properties
    print('\nMerging catalogs...')
    final_df = merge_agn_with_halos(agn_triples_df, halo_df)

    # Save final catalog
    print('\nSaving final catalog...')
    save_triple_agn_catalog(final_df, output_path, z)

    print("\n" + "="*60)
    print("COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    print('Starting triple AGN catalog builder')
    print('='*60)
    parser = argparse.ArgumentParser(
        description="Build triple AGN catalog from Romulus25 simulation"
    )
    parser.add_argument(
        "snapshot_index", 
        type=int, 
        help="Index in snapshot list to process"
    )
    args = parser.parse_args()
    main(snapshot_index=args.snapshot_index)

print('Finished')