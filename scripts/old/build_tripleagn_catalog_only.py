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
#STEP 3: Match BHs between Tangos and Pynbody
##############################################
def match_bh_properties(sim, tangos_df):
    print('step 3')
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
        'BH_x': matched_pos[:, 0],
        'BH_y': matched_pos[:, 1],
        'BH_z': matched_pos[:, 2],
    })
    
    merged = pd.merge(tangos_df, matched_df, on='BH_id', how='inner')
    return merged

##############################################
#STEP 4: Compute BH local environment properties
##############################################
def compute_bh_local_env(sim, pos, t_now, t_prev):
    print('step 4')
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
#STEP 5: Identify triple AGNs - CORRECTED VERSION
##############################################
def find_triple_agn_systems(df, sim, t_now, t_prev, separation_threshold=100, mdot_threshold=1.89e-3):
    print('step 5')

    # Compute Lbol and Ledd for ALL BHs upfront
    df = df.copy()

    msun_to_g = 1.989e33
    yr_to_s = 3.154e7
    c_cgs = 3e10

    df['BH_Lbol'] = 0.1 * df['BH_mdot'] * msun_to_g / yr_to_s * c_cgs**2
    df['BH_Ledd'] = 1.26e38 * df['BH_mass']

    # Select only BHs that meet the AGN threshold
    main_bhs = df[df['BH_Lbol'] > 1e40].copy()
    triples = []

    for _, bh in main_bhs.iterrows():
        bh_pos = np.array([bh['BH_x'], bh['BH_y'], bh['BH_z']])
        deltas = df[['BH_x', 'BH_y', 'BH_z']].values - bh_pos
        distances = np.linalg.norm(deltas, axis=1)

        # Select neighbor BHs that are also AGN and not the same as current BH
        neighbors = df[(distances < separation_threshold) & 
                       (df['BH_mdot'] > mdot_threshold) & 
                       (df['BH_id'] != bh['BH_id'])]

        # CRITICAL: Only accept systems with EXACTLY 2 neighbors (not more, not less)
        if len(neighbors) > 2:
            # Skip this system - too many neighbors
            continue
        elif len(neighbors) == 2:
            # This is a valid triple AGN system
            neighbor1 = neighbors.iloc[0]
            neighbor2 = neighbors.iloc[1]

            # Calculate separations for neighbor 1
            dx1 = bh['BH_x'] - neighbor1['BH_x']
            dy1 = bh['BH_y'] - neighbor1['BH_y']
            dz1 = bh['BH_z'] - neighbor1['BH_z']
            separation_3d_1 = np.sqrt(dx1**2 + dy1**2 + dz1**2)
            separation_2d_1 = np.sqrt(dx1**2 + dy1**2)

            # Calculate separations for neighbor 2
            dx2 = bh['BH_x'] - neighbor2['BH_x']
            dy2 = bh['BH_y'] - neighbor2['BH_y']
            dz2 = bh['BH_z'] - neighbor2['BH_z']
            separation_3d_2 = np.sqrt(dx2**2 + dy2**2 + dz2**2)
            separation_2d_2 = np.sqrt(dx2**2 + dy2**2)

            # Compute environments for main BH
            bh_env = compute_bh_local_env(sim, bh_pos, t_now, t_prev)
            
            # Compute environments for neighbor 1
            neighbor1_env = compute_bh_local_env(
                sim,
                [neighbor1['BH_x'], neighbor1['BH_y'], neighbor1['BH_z']],
                t_now,
                t_prev
            )
            neighbor1_env = {f'Neighbour1{key}': val for key, val in neighbor1_env.items()}

            # Compute environments for neighbor 2
            neighbor2_env = compute_bh_local_env(
                sim,
                [neighbor2['BH_x'], neighbor2['BH_y'], neighbor2['BH_z']],
                t_now,
                t_prev
            )
            neighbor2_env = {f'Neighbour2{key}': val for key, val in neighbor2_env.items()}

            # Prepare neighbor 1 data
            neighbor1_dict = {f'Neighbour1{col}': neighbor1[col] for col in df.columns if col.startswith('BH_') or col == 'BH_id'}
            neighbor1_dict.update({
                'Separation_3D_kpc_1': separation_3d_1,
                'Separation_2D_kpc_1': separation_2d_1,
                'Neighbour1BH_pynbody_haloid': neighbor1['pynbody_haloid'],
                'Neighbour1BH_tangos_haloid': neighbor1['tangos_haloid']
            })

            # Prepare neighbor 2 data
            neighbor2_dict = {f'Neighbour2{col}': neighbor2[col] for col in df.columns if col.startswith('BH_') or col == 'BH_id'}
            neighbor2_dict.update({
                'Separation_3D_kpc_2': separation_3d_2,
                'Separation_2D_kpc_2': separation_2d_2,
                'Neighbour2BH_pynbody_haloid': neighbor2['pynbody_haloid'],
                'Neighbour2BH_tangos_haloid': neighbor2['tangos_haloid']
            })

            triple = {**bh.to_dict(), **bh_env, **neighbor1_dict, **neighbor1_env, **neighbor2_dict, **neighbor2_env}
            triples.append(triple)

            print('Found a triple!')

    return pd.DataFrame(triples)

##############################################
# STEP 6: Compute halo properties - FIXED VERSION
##############################################
def compute_halo_environment_properties(sim, halo_id_pairs, t_now, t_prev):
    print('step6')
    
    # Load AHF halo catalog
    try:
        h = sim.halos(ahf_mpi=True)
        print(f'  Loaded {len(h)} halos from AHF catalog')
    except Exception as e:
        print(f'  ERROR: Failed to load halo catalog: {e}')
        return pd.DataFrame()
    
    # Filter valid halo IDs
    valid_pairs = [(pnb_id, tng_id) for pnb_id, tng_id in halo_id_pairs 
                   if pnb_id > 0 and tng_id > 0 and not np.isnan(pnb_id) and not np.isnan(tng_id)]
    
    if len(valid_pairs) == 0:
        print("  WARNING: No valid halo IDs to process")
        return pd.DataFrame()
    
    print(f'  Processing {len(valid_pairs)} halo ID pairs')
    
    # Get unique pairs
    unique_pairs = list(set(valid_pairs))
    print(f'  ({len(unique_pairs)} unique halos)')
    
    results = []
    successful = 0
    failed = 0
    
    for pnb_haloid, tng_haloid in unique_pairs:
        try:
            # Use tangos_haloid as the array index
            halo_idx = int(tng_haloid)
            halo = h[halo_idx]
            
            center = pnb.analysis.halo.center(halo, mode='pot', retcen=True).in_units('kpc')
            virovdens = cos.Delta_vir(sim)
            rdict = {'vir': virovdens, '200': 200, '500': 500, '2500': 2500}
            MassRadii = cR.get_radius(halo, list(rdict.values()), prop=sim.properties, cen=center)

            def compute_sfr(sub):
                newstars = sub.s['tform'] > t_prev
                return sub.s['mass'][newstars].sum() / ((t_now - t_prev) * 1e9) if len(sub.s['mass'][newstars]) > 0 else 0.0

            sub25 = halo[pnb.filt.Sphere(25, center)]
            sub30 = halo[pnb.filt.Sphere(30, center)]
            sub50 = halo[pnb.filt.Sphere(50, center)]

            def half_mass_radius():
                try:
                    radii = np.linspace(0.1, 0.3 * MassRadii[1][200.0], 100)
                    masses = []
                    for r in radii:
                        mass = halo[pnb.filt.Sphere(r, center)].s['mass'].sum()
                        masses.append(mass)
                    masses = np.array(masses)
                    total_mass = masses[-1]
                    if total_mass == 0:
                        return np.nan
                    half_mass = total_mass / 2
                    idx = np.where(masses >= half_mass)[0]
                    if len(idx) == 0:
                        return np.nan
                    return 4.0 * radii[idx[0]]
                except:
                    return np.nan

            Rgal = half_mass_radius()
            
            if not np.isnan(Rgal) and Rgal > 0:
                subgal = halo[pnb.filt.Sphere(Rgal, center)]
            else:
                subgal = sub50

            results.append({
                'Halo_id': int(pnb_haloid),
                'Halo_tangos_id': int(tng_haloid),
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
            
            successful += 1
            
        except Exception as e:
            print(f'  Failed halo (pnb={int(pnb_haloid)}, tng={int(tng_haloid)}): {e}')
            failed += 1
            continue
    
    print(f'  Successfully processed: {successful}/{len(unique_pairs)} halos')
    print(f'  Failed: {failed}/{len(unique_pairs)} halos')
    
    if len(results) == 0:
        print("  WARNING: No halos were successfully processed")
        return pd.DataFrame()
    
    return pd.DataFrame(results).drop_duplicates('Halo_id').set_index('Halo_id')

##############################################
#STEP 7: Merge AGN catalog with halo properties
##############################################
def merge_agn_with_halos(agn_df, halo_df):
    print('step 7')
    df = pd.merge(agn_df, halo_df, how='left', left_on='pynbody_haloid', right_index=True)
    df = pd.merge(df, halo_df.add_prefix('Neighbour1Halo_'), how='left', left_on='Neighbour1BH_pynbody_haloid', right_index=True)
    df = pd.merge(df, halo_df.add_prefix('Neighbour2Halo_'), how='left', left_on='Neighbour2BH_pynbody_haloid', right_index=True)
    return df

##############################################
#STEP 8: Save final catalog
##############################################
def save_triple_agn_catalog(df, output_path_template, z):
    print('step 8')

    base = output_path_template.format(z)
    csv_path = base.replace(".pkl", ".csv")
    h5_path = base.replace(".pkl", ".h5")
    pkl_path = base

    df.to_pickle(pkl_path)
    df.to_csv(csv_path, index=False)
    df.to_hdf(h5_path, key="tripleagn", mode="w")

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
            '001302','001378','001458','001536','001543','001550','001632','001726','001792','001826','001931','001945',\
            '002042','002048','002159','002281','002304','002411','002536','002547','002560','002690','002816','002840',\
            '002998','003072','003163','003328','003336','003478','003517','003584','003707','003840','003905','004096',\
            '004111','004173','004326','004352','004549','004608','004781','004864','005022','005107','005120','005271',\
            '005376','005529','005632','005795','005888','006069','006144','006350','006390','006400','006640','006656',\
            '006912','006937','007168','007212','007241','007394','007424','007552','007680','007779','007869','007936',\
            '008192']  
    hdf5_path = "/scratch/stlock/tripleAGNs/halomap_files/HaloBH-TangosPynbodyMap-R25-snap{}.hdf5"
    output_path = "/scratch/stlock/tripleAGNs/datasets/catalogue_only/TripleAGN-Catalog-R100-z{:.2f}.pkl"
    sim_path = "/home/stlock/projects/rrg-babul-ad/SHARED/Romulus/cosmo25/"

    print('loading snapshots')
    s, snap, z, t_now = load_snapshot(snapshot_index, snapshot_list, sim_path)
    s_last, _, _, t_prev = load_snapshot(snapshot_index - 1, snapshot_list, sim_path)
    print('loading tangos')
    tangos_df = load_tangos_bh_data(snap, hdf5_path)
    print('calculating bh properties')
    matched_bh_df = match_bh_properties(s, tangos_df)
    print('find triple agn systems')
    agn_triples_df = find_triple_agn_systems(matched_bh_df, s, t_now, t_prev)

    # Check if any triple systems were found
    if len(agn_triples_df) == 0:
        print(f'No triple AGN systems found for snapshot {snap} at z={z:.2f}')
        print('Skipping this snapshot.')
        return

    # Get halo ID pairs for all BHs in triples
    halo_id_pairs = []
    for _, row in agn_triples_df.iterrows():
        halo_id_pairs.append((row['pynbody_haloid'], row['tangos_haloid']))
        halo_id_pairs.append((row['Neighbour1BH_pynbody_haloid'], row['Neighbour1BH_tangos_haloid']))
        halo_id_pairs.append((row['Neighbour2BH_pynbody_haloid'], row['Neighbour2BH_tangos_haloid']))
    
    halo_id_pairs = list(set(halo_id_pairs))
    
    # Compute halo properties
    halo_df = compute_halo_environment_properties(s, halo_id_pairs, t_now, t_prev)

    if len(halo_df) == 0:
        print(f'WARNING: No halo properties could be computed for snapshot {snap} at z={z:.2f}')
        print('Saving AGN catalog without halo information.')
        final_df = agn_triples_df
    else:
        print(f'Successfully computed properties for {len(halo_df)} halos')
        final_df = merge_agn_with_halos(agn_triples_df, halo_df)
    
    save_triple_agn_catalog(final_df, output_path, z)
    print(f'Successfully saved catalog with {len(final_df)} triple AGN systems')

if __name__ == "__main__":
    print('start running the main')
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_index", type=int, help="Index in snapshot list to process")
    args = parser.parse_args()
    main(snapshot_index=args.snapshot_index)

print('Finished')