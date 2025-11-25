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


##############################################
#STEP 1: Load snapshot and redshift info
##############################################
def load_snapshot(snapshot_index, snapshot_list, base_path):
    print('step 1')
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
    print('step 2')
    with h5py.File(hdf5_path.format(snap), 'r') as hf:
        data = {k: np.array(hf[k]) for k in hf.keys()}
    return pd.DataFrame(data)

##############################################
#print('STEP 3: Match BHs between Tangos and Pynbody')
##############################################
def match_bh_properties(sim, tangos_df, z):
    print('step 3')
    BH_idx = np.where(sim.s['tform'] < 0)
    all_bh_ids = sim.s['iord'][BH_idx]
    match_mask = np.isin(all_bh_ids, tangos_df['BH_id'].values)

    matched_ids = sim.s['iord'][BH_idx][match_mask]
    matched_vel = sim.s['vel'][BH_idx][match_mask]
    # positions in physical kpc from snapshot, convert to comoving kpc
    matched_pos_phys = sim.s['pos'][BH_idx][match_mask].in_units('kpc')
    matched_pos_com = matched_pos_phys * (1.0 + z)

    matched_df = pd.DataFrame({
        'BH_id': matched_ids,
        'BH_vx': matched_vel[:, 0],
        'BH_vy': matched_vel[:, 1],
        'BH_vz': matched_vel[:, 2],
        'BH_x': matched_pos_com[:, 0],
        'BH_y': matched_pos_com[:, 1],
        'BH_z': matched_pos_com[:, 2],
    })
    
    merged = pd.merge(tangos_df, matched_df, on='BH_id', how='inner')
    return merged

##############################################
#print('STEP 4: Compute BH local environment properties')
##############################################
def compute_bh_local_env(sim, pos_comoving, t_now, t_prev, z):
    print('step 4')
    result = {}
    a = 1.0 / (1.0 + z)  # scale factor
    # convert comoving center -> physical for selection with snapshot (snapshot is in physical units)
    center_phys = np.array(pos_comoving) * a

    for r_com in [1, 5]:  # radii now interpreted as comoving kpc
        # convert to physical radius for selection
        r_phys = r_com * a
        region = sim[pnb.filt.Sphere(r_phys, center_phys)]
        mstar = region.s['mass'].sum()
        mgas = region.gas['mass'].sum()
        # compute volume in comoving kpc^3 (use comoving r for density)
        volume_com = (4/3) * np.pi * (r_com**3)

        result[f'BH_Mstar_{r_com}'] = mstar
        result[f'BH_GasDensity_{r_com}'] = mgas / volume_com if volume_com > 0 else np.nan

        if len(region.gas) > 0:
            gas_mass = region.gas['mass']
            gas_metal = region.gas['metals'] * gas_mass
            result[f'BH_GasMetallicity_{r_com}'] = gas_metal.sum() / gas_mass.sum() if gas_mass.sum() > 0 else np.nan
        else:
            result[f'BH_GasMetallicity_{r_com}'] = np.nan

        if len(region.s) > 0:
            star_mass = region.s['mass']
            star_metal = region.s['metals'] * star_mass
            result[f'BH_StellarMetallicity_{r_com}'] = star_metal.sum() / star_mass.sum() if star_mass.sum() > 0 else np.nan
        else:
            result[f'BH_StellarMetallicity_{r_com}'] = np.nan

        newstars = region.s['tform'] > t_prev
        # SFR reported in Msun/yr using comoving volume/time window
        result[f'BH_SFR_{r_com}'] = region.s['mass'][newstars].sum() / ((t_now - t_prev) * 1e9) if len(region.s['mass'][newstars]) > 0 else 0

    return result

##############################################
#print('STEP 5: Identify triple AGNs')
##############################################
def find_triple_agn_systems(df, sim, t_now, t_prev, separation_threshold=50, mdot_threshold=1.89e-3, z=None):
    print('step 5: Finding triple AGN systems')

    # --- Compute Lbol and Ledd for ALL BHs upfront ---
    df = df.copy()  # to safely add new columns

    msun_to_g = 1.989e33
    yr_to_s = 3.154e7
    c_cgs = 3e10

    # Compute and store Lbol and Ledd
    df['BH_Lbol'] = 0.1 * df['BH_mdot'] * msun_to_g / yr_to_s * c_cgs**2
    df['BH_Ledd'] = 1.26e38 * df['BH_mass']

    # --- Select only BHs that meet the AGN threshold ---
    agn_bhs = df[df['BH_Lbol'] > 1e43].copy()
    
    print(f"Total AGNs found: {len(agn_bhs)}")
    
    # Track which BHs have been assigned to a system
    assigned_bhs = set()
    triple_systems = []

    # Iterate through all AGN BHs to find triple systems
    for idx, bh in agn_bhs.iterrows():
        # Skip if this BH is already in a system
        if bh['BH_id'] in assigned_bhs:
            continue
            
        # BH positions in the dataframe are now in comoving kpc
        bh_pos_com = np.array([bh['BH_x'], bh['BH_y'], bh['BH_z']])
        
        # Calculate distances to all other BHs in comoving kpc
        deltas = df[['BH_x', 'BH_y', 'BH_z']].values - bh_pos_com
        distances = np.linalg.norm(deltas, axis=1)

        # Select neighbor BHs that are also AGN, not assigned, and not the current BH
        neighbors = df[(distances < separation_threshold) & 
                       (df['BH_Lbol'] > 1e43) &
                       (df['BH_mdot'] > mdot_threshold) & 
                       (df['BH_id'] != bh['BH_id']) &
                       (~df['BH_id'].isin(assigned_bhs))]

        # Check if we have exactly 2 neighbors (triple system)
        if len(neighbors) == 2:
            print(f'Triple AGN system detected around BH {bh["BH_id"]}')
            
            neighbor1 = neighbors.iloc[0]
            neighbor2 = neighbors.iloc[1]
            
            # Mark all three BHs as assigned
            assigned_bhs.add(bh['BH_id'])
            assigned_bhs.add(neighbor1['BH_id'])
            assigned_bhs.add(neighbor2['BH_id'])
            
            # Compute separations in comoving kpc
            dx1 = bh['BH_x'] - neighbor1['BH_x']
            dy1 = bh['BH_y'] - neighbor1['BH_y']
            dz1 = bh['BH_z'] - neighbor1['BH_z']
            separation_3d_1 = np.sqrt(dx1**2 + dy1**2 + dz1**2)
            separation_2d_1 = np.sqrt(dx1**2 + dy1**2)
            
            dx2 = bh['BH_x'] - neighbor2['BH_x']
            dy2 = bh['BH_y'] - neighbor2['BH_y']
            dz2 = bh['BH_z'] - neighbor2['BH_z']
            separation_3d_2 = np.sqrt(dx2**2 + dy2**2 + dz2**2)
            separation_2d_2 = np.sqrt(dx2**2 + dy2**2)
            
            # Separation between the two neighbors
            dx12 = neighbor1['BH_x'] - neighbor2['BH_x']
            dy12 = neighbor1['BH_y'] - neighbor2['BH_y']
            dz12 = neighbor1['BH_z'] - neighbor2['BH_z']
            separation_3d_12 = np.sqrt(dx12**2 + dy12**2 + dz12**2)
            separation_2d_12 = np.sqrt(dx12**2 + dy12**2)
            
            # Compute local environments: pass comoving BH positions and redshift
            bh_env = compute_bh_local_env(sim, bh_pos_com, t_now, t_prev, z)
            
            neighbor1_env = compute_bh_local_env(
                sim,
                np.array([neighbor1['BH_x'], neighbor1['BH_y'], neighbor1['BH_z']]),
                t_now,
                t_prev,
                z
            )
            neighbor1_env = {f'Neighbour1{key}': val for key, val in neighbor1_env.items()}
            
            neighbor2_env = compute_bh_local_env(
                sim,
                np.array([neighbor2['BH_x'], neighbor2['BH_y'], neighbor2['BH_z']]),
                t_now,
                t_prev,
                z
            )
            neighbor2_env = {f'Neighbour2{key}': val for key, val in neighbor2_env.items()}
            
            # Create dictionaries for neighbors
            neighbor1_dict = {f'Neighbour1{col}': neighbor1[col] for col in df.columns 
                            if col.startswith('BH_') or col == 'BH_id'}
            neighbor1_dict.update({
                'Separation_3D_1_kpc_com': separation_3d_1,
                'Separation_2D_1_kpc_com': separation_2d_1,
                'Neighbour1BH_pynbody_haloid': neighbor1['pynbody_haloid']
            })
            
            neighbor2_dict = {f'Neighbour2{col}': neighbor2[col] for col in df.columns 
                            if col.startswith('BH_') or col == 'BH_id'}
            neighbor2_dict.update({
                'Separation_3D_2_kpc_com': separation_3d_2,
                'Separation_2D_2_kpc_com': separation_2d_2,
                'Neighbour2BH_pynbody_haloid': neighbor2['pynbody_haloid']
            })
            
            # Add separation between neighbors
            cross_separation_dict = {
                'Separation_3D_12_kpc_com': separation_3d_12,
                'Separation_2D_12_kpc_com': separation_2d_12
            }
            
            # Combine all data
            triple_system = {**bh.to_dict(), **bh_env, 
                           **neighbor1_dict, **neighbor1_env,
                           **neighbor2_dict, **neighbor2_env,
                           **cross_separation_dict}
            triple_systems.append(triple_system)
            
        elif len(neighbors) > 2:
            print(f'More than 2 AGN neighbors ({len(neighbors)}) detected near BH {bh["BH_id"]} - skipping')

    print(f"Total triple AGN systems found: {len(triple_systems)}")
    return pd.DataFrame(triple_systems)

##############################################
# STEP 6: Compute halo properties (main and neighbors)
##############################################
def compute_halo_environment_properties(sim, halo_ids, t_now, t_prev, z):
    print('step6')
    h = sim.halos(ahf_mpi=True)
    results = []

    # sanitize halo_ids: drop NaN, ensure integers and unique
    try:
        halo_ids_arr = np.asarray(halo_ids)
    except Exception:
        halo_ids_arr = np.array(list(halo_ids))

    # drop NaNs and convert to ints
    halo_ids_clean = []
    for x in np.unique(halo_ids_arr):
        if pd.isna(x):
            continue
        try:
            halo_ids_clean.append(int(x))
        except Exception:
            continue

    a = 1.0 / (1.0 + z)  # scale factor

    for haloid in halo_ids_clean:
        try:
            # center in physical kpc
            center_phys = pnb.analysis.halo.center(h[haloid], mode='pot', retcen=True).in_units('kpc')
            virovdens = cos.Delta_vir(sim)
            rdict = {'vir': virovdens, '200': 200, '500': 500, '2500': 2500}
            MassRadii = cR.get_radius(h[haloid], list(rdict.values()), prop=sim.properties, cen=center_phys)

            def compute_sfr(sub):
                newstars = sub.s['tform'] > t_prev
                return sub.s['mass'][newstars].sum() / ((t_now - t_prev) * 1e9)

            # select subregions: radii provided are comoving (25,30,50 comoving kpc)
            sub25 = h[haloid][pnb.filt.Sphere(25 * a, center_phys)]
            sub30 = h[haloid][pnb.filt.Sphere(30 * a, center_phys)]
            sub50 = h[haloid][pnb.filt.Sphere(50 * a, center_phys)]

            def half_mass_radius():
                # MassRadii[1][200.0] is physical; search in physical radii
                radii = np.linspace(0.1, 0.3 * MassRadii[1][200.0], 100)
                masses = []
                for r in radii:
                    mass = h[haloid][pnb.filt.Sphere(r, center_phys)].s['mass'].sum()
                    masses.append(mass)
                masses = np.array(masses)
                total_mass = masses[-1]
                half_mass = total_mass / 2
                idx = np.where(masses >= half_mass)[0][0]
                return 4.0 * radii[idx]  # returns physical kpc

            Rgal_phys = half_mass_radius()
            subgal = h[haloid][pnb.filt.Sphere(Rgal_phys, center_phys)]

            # convert centers and radii to comoving for output (comoving = physical * (1+z))
            center_com = center_phys * (1.0 + z)

            results.append({
                'Halo_id': int(haloid),
                'Halo_center_x': center_com[0],
                'Halo_center_y': center_com[1],
                'Halo_center_z': center_com[2],
                'Halo_Mvir': MassRadii[0][virovdens],
                'Halo_Rvir': MassRadii[1][virovdens] * (1.0 + z),
                'Halo_M200': MassRadii[0][200.0],
                'Halo_R200': MassRadii[1][200.0] * (1.0 + z),
                'Halo_M500': MassRadii[0][500.0],
                'Halo_R500': MassRadii[1][500.0] * (1.0 + z),
                'Halo_M2500': MassRadii[0][2500.0],
                'Halo_R2500': MassRadii[1][2500.0] * (1.0 + z),
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
                'Halo_Rgal': Rgal_phys * (1.0 + z)  # report comoving
            })

        except Exception as e:
            print(f"Failed halo {haloid}: {e}")

    df = pd.DataFrame(results)

    if df.empty:
        # return empty dataframe with expected index name
        return pd.DataFrame(columns=['Halo_id']).set_index('Halo_id')
    else:
        return df.drop_duplicates('Halo_id').set_index('Halo_id')

        
##############################################
#'STEP 7: Merge AGN catalog with halo properties (main + 2 neighbors)')
##############################################
def merge_agn_with_halos(agn_df, halo_df):
    print('step 7')
    # Merge main BH halo properties
    df = pd.merge(agn_df, halo_df, how='left', left_on='pynbody_haloid', right_index=True)
    
    # Merge neighbor 1 halo properties
    df = pd.merge(df, halo_df.add_prefix('Neighbour1Halo_'), how='left', 
                  left_on='Neighbour1BH_pynbody_haloid', right_index=True)
    
    # Merge neighbor 2 halo properties
    df = pd.merge(df, halo_df.add_prefix('Neighbour2Halo_'), how='left', 
                  left_on='Neighbour2BH_pynbody_haloid', right_index=True)
    
    return df

##############################################
#print('STEP 8: Save final catalog')
##############################################
def save_triple_agn_catalog(df, output_path_template, z):
    print('step 8')

    # Format file paths
    base = output_path_template.format(z)
    csv_path = base.replace(".pkl", ".csv")
    h5_path = base.replace(".pkl", ".h5")
    pkl_path = base

    # Save files
    df.to_pickle(pkl_path)  # for internal use
    df.to_csv(csv_path, index=False)  # for collaborators
    df.to_hdf(h5_path, key="tripleagn", mode="w")  # for structured access

    print(f"Saved catalog for redshift {z:.2f} to:")
    print(f" - Pickle: {pkl_path}")
    print(f" - CSV:    {csv_path}")
    print(f" - HDF5:   {h5_path}")

##############################################
# MAIN EXECUTION FLOW
##############################################
def main(snapshot_index):
    snapshot_list = ['000098','000105','000111','000118','000126','000134','000142','000151','000160','000170','000181',\
            '000192','000204','000216','000229','000243','000256','000258','000274','000290','000308',\
            '000372','000437','000446',\
            '000463','000491','000512','000520','000547','000551','000584','000618','000655',\
            '000690','000694','000735','000768','000778','000824','000873','000909','000924','000979','001024',\
            '001036','001065','001097','001162','001230','001270','001280',\
            '001302','001378','001458','001536','001543','001550','001632','001726','01792','001826','001931','001945',\
            '002042','002048','002159','002281','002304','002411','002536','002547','002560','002690','002816','002840',\
            '002998','003072','003163','003328','003336','003478','003517','003584','003707','003840','003905','004096',\
            '004111','004173','004326','004352','004549','004608','004781','004864','005022','005107','005120','005271',\
            '005376','005529','005632','005795','005888','006069','006144','006350','006390','006400','006640','006656',\
            '006912','006937','007168','007212','007241','007394','007424','007552','007680','007779','007869','007936',\
            '008192']  
    hdf5_path = "/scratch/stlock/halomap_files/HaloBH-TangosPynbodyMap-R25-snap{}.hdf5"
    output_path = "/scratch/stlock/tripleAGNs/catalogs/1e43lum/catalogue_50kpc_1e43lum/TripleAGN-Catalog-R50-z{:.2f}.pkl"
    sim_path = "/home/stlock/projects/rrg-babul-ad/SHARED/Romulus/cosmo25/"

    print('loading snapshots')
    s, snap, z, t_now = load_snapshot(snapshot_index, snapshot_list, sim_path)
    s_last, _, _, t_prev = load_snapshot(snapshot_index - 1, snapshot_list, sim_path)
    print('loading tangos')
    tangos_df = load_tangos_bh_data(snap, hdf5_path)
    print('calculating bh properties')
    # changed code: pass `z` into functions and handle empty result
    matched_bh_df = match_bh_properties(s, tangos_df, z)         # positions -> comoving using z
    print('find triple agn systems')
    triple_agn_df = find_triple_agn_systems(matched_bh_df, s, t_now, t_prev, z=z)

    if len(triple_agn_df) == 0:
        print(f"No triple AGN systems found at z={z:.2f}")
        save_triple_agn_catalog(triple_agn_df, output_path, z)
        return

    # collect halo ids and compute halo properties (pass z)
    all_halo_ids = pd.unique(triple_agn_df[['pynbody_haloid',
                                             'Neighbour1BH_pynbody_haloid',
                                             'Neighbour2BH_pynbody_haloid']].values.ravel())

    halo_df = compute_halo_environment_properties(s, all_halo_ids, t_now, t_prev, z)
    final_df = merge_agn_with_halos(triple_agn_df, halo_df)
    save_triple_agn_catalog(final_df, output_path, z)

if __name__ == "__main__":
    print('start running the main')
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_index", type=int, help="Index in snapshot list to process")
    args = parser.parse_args()
    main(snapshot_index=args.snapshot_index)

print('Finished')