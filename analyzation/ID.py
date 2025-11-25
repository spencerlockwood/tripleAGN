import numpy as np
import pandas as pd
import glob
import os

# Define paths
catalog_path = "/scratch/stlock/tripleAGNs/catalogs/1e43lum/catalogue_30kpc_1e43lum/"
output_dir = "/home/stlock/tripleAGN/analysis/"
os.makedirs(output_dir, exist_ok=True)

print("Extracting Triple AGN System IDs...")

# Search for catalog files
files_pkl = sorted(glob.glob(f"{catalog_path}TripleAGN-Catalog-R50-z*.pkl"))
files_csv = sorted(glob.glob(f"{catalog_path}TripleAGN-Catalog-R50-z*.csv"))
files_h5 = sorted(glob.glob(f"{catalog_path}TripleAGN-Catalog-R50-z*.h5"))

# Prefer pickle files, then CSV, then HDF5
if files_pkl:
    files_to_read = files_pkl
    file_format = 'pkl'
elif files_csv:
    files_to_read = files_csv
    file_format = 'csv'
elif files_h5:
    files_to_read = files_h5
    file_format = 'h5'
else:
    print("No catalog files found!")
    exit(1)

print(f"Found {len(files_to_read)} catalog files ({file_format} format)")

# Storage for all systems across redshifts
all_systems = []

# Read each file
for file_path in files_to_read:
    try:
        # Read the file based on format
        if file_format == 'pkl':
            df = pd.read_pickle(file_path)
        elif file_format == 'csv':
            df = pd.read_csv(file_path)
        elif file_format == 'h5':
            df = pd.read_hdf(file_path, key='tripleagn')
        
        if len(df) == 0:
            continue
        
        # Extract redshift from filename
        filename = os.path.basename(file_path)
        z_str = filename.split('-z')[1].split('.')[0] + '.' + filename.split('-z')[1].split('.')[1]
        z = float(z_str)
        
        print(f"\nz={z:.2f}: {len(df)} triple AGN system(s)")
        
        # Extract relevant ID columns for each system
        for idx, row in df.iterrows():
            system_info = {
                'redshift': z,
                'system_number': idx,
                
                # Primary BH
                'BH1_id': row['BH_id'],
                'BH1_halo_pynbody': row['pynbody_haloid'],
                'BH1_mass': row['BH_mass'],
                'BH1_mdot': row['BH_mdot'],
                'BH1_Lbol': row['BH_Lbol'],
                'BH1_x': row['BH_x'],
                'BH1_y': row['BH_y'],
                'BH1_z': row['BH_z'],
                
                # Neighbor 1 BH
                'BH2_id': row['Neighbour1BH_id'],
                'BH2_halo_pynbody': row['Neighbour1BH_pynbody_haloid'],
                'BH2_mass': row['Neighbour1BH_mass'],
                'BH2_mdot': row['Neighbour1BH_mdot'],
                'BH2_Lbol': row['Neighbour1BH_Lbol'],
                'BH2_x': row['Neighbour1BH_x'],
                'BH2_y': row['Neighbour1BH_y'],
                'BH2_z': row['Neighbour1BH_z'],
                
                # Neighbor 2 BH
                'BH3_id': row['Neighbour2BH_id'],
                'BH3_halo_pynbody': row['Neighbour2BH_pynbody_haloid'],
                'BH3_mass': row['Neighbour2BH_mass'],
                'BH3_mdot': row['Neighbour2BH_mdot'],
                'BH3_Lbol': row['Neighbour2BH_Lbol'],
                'BH3_x': row['Neighbour2BH_x'],
                'BH3_y': row['Neighbour2BH_y'],
                'BH3_z': row['Neighbour2BH_z'],
                
                # Separations
                'separation_BH1_BH2_kpc': row['Separation_3D_1_kpc'],
                'separation_BH1_BH3_kpc': row['Separation_3D_2_kpc'],
                'separation_BH2_BH3_kpc': row['Separation_3D_12_kpc'],
                
                # Halo masses (if available)
                'Halo1_Mvir': row.get('Halo_Mvir', np.nan),
                'Halo1_M200': row.get('Halo_M200', np.nan),
                'Halo2_Mvir': row.get('Neighbour1Halo_Mvir', np.nan),
                'Halo2_M200': row.get('Neighbour1Halo_M200', np.nan),
                'Halo3_Mvir': row.get('Neighbour2Halo_Mvir', np.nan),
                'Halo3_M200': row.get('Neighbour2Halo_M200', np.nan),
            }
            
            all_systems.append(system_info)
            
            # Print detailed info for this system
            print(f"  System {idx}:")
            print(f"    BH1: id={system_info['BH1_id']:.0f}, halo={system_info['BH1_halo_pynbody']:.0f}, M={system_info['BH1_mass']:.2e} Msun")
            print(f"    BH2: id={system_info['BH2_id']:.0f}, halo={system_info['BH2_halo_pynbody']:.0f}, M={system_info['BH2_mass']:.2e} Msun")
            print(f"    BH3: id={system_info['BH3_id']:.0f}, halo={system_info['BH3_halo_pynbody']:.0f}, M={system_info['BH3_mass']:.2e} Msun")
            print(f"    Separations: {system_info['separation_BH1_BH2_kpc']:.1f}, {system_info['separation_BH1_BH3_kpc']:.1f}, {system_info['separation_BH2_BH3_kpc']:.1f} kpc")
            
            # Check if all three BHs are in the same halo
            same_halo = (system_info['BH1_halo_pynbody'] == system_info['BH2_halo_pynbody'] == system_info['BH3_halo_pynbody'])
            if same_halo:
                print(f"    ** All 3 BHs in same halo (ID={system_info['BH1_halo_pynbody']:.0f}) **")
            else:
                unique_halos = len(set([system_info['BH1_halo_pynbody'], 
                                       system_info['BH2_halo_pynbody'], 
                                       system_info['BH3_halo_pynbody']]))
                print(f"    ** BHs distributed across {unique_halos} different halo(s) **")
        
    except Exception as e:
        print(f"Error reading {os.path.basename(file_path)}: {e}")
        continue

if len(all_systems) == 0:
    print("\nNo triple AGN systems found in any snapshot!")
    exit(0)

# Convert to DataFrame
systems_df = pd.DataFrame(all_systems)

# Save to multiple formats
csv_path = os.path.join(output_dir, "triple_agn_system_ids.csv")
#pkl_path = os.path.join(output_dir, "triple_agn_system_ids.pkl")
#h5_path = os.path.join(output_dir, "triple_agn_system_ids.h5")

systems_df.to_csv(csv_path, index=False)
#systems_df.to_pickle(pkl_path)
#systems_df.to_hdf(h5_path, key='systems', mode='w')

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total triple AGN systems found: {len(systems_df)}")
print(f"Redshift range: {systems_df['redshift'].min():.2f} - {systems_df['redshift'].max():.2f}")
print(f"Number of unique redshifts with systems: {systems_df['redshift'].nunique()}")

# Analyze halo configurations
systems_df['num_unique_halos'] = systems_df.apply(
    lambda row: len(set([row['BH1_halo_pynbody'], row['BH2_halo_pynbody'], row['BH3_halo_pynbody']])), 
    axis=1
)

print(f"\nHalo configuration breakdown:")
print(f"  All 3 BHs in same halo: {(systems_df['num_unique_halos'] == 1).sum()}")
print(f"  BHs in 2 different halos: {(systems_df['num_unique_halos'] == 2).sum()}")
print(f"  BHs in 3 different halos: {(systems_df['num_unique_halos'] == 3).sum()}")

print(f"\nAverage separations (kpc):")
print(f"  BH1-BH2: {systems_df['separation_BH1_BH2_kpc'].mean():.2f} ± {systems_df['separation_BH1_BH2_kpc'].std():.2f}")
print(f"  BH1-BH3: {systems_df['separation_BH1_BH3_kpc'].mean():.2f} ± {systems_df['separation_BH1_BH3_kpc'].std():.2f}")
print(f"  BH2-BH3: {systems_df['separation_BH2_BH3_kpc'].mean():.2f} ± {systems_df['separation_BH2_BH3_kpc'].std():.2f}")

print(f"\nFiles saved:")
print(f"  CSV:   {csv_path}")
#print(f"  PKL:   {pkl_path}")
#print(f"  HDF5:  {h5_path}")

# Create a simple reference table
print("\n" + "="*70)
print("QUICK REFERENCE TABLE")
print("="*70)
print(systems_df[['redshift', 'BH1_id', 'BH2_id', 'BH3_id', 
                   'BH1_halo_pynbody', 'BH2_halo_pynbody', 'BH3_halo_pynbody',
                   'num_unique_halos']].to_string(index=False))

# Save the reference table as well
ref_table_path = os.path.join(output_dir, "triple_agn_quick_reference.txt")
with open(ref_table_path, 'w') as f:
    f.write("TRIPLE AGN SYSTEMS - QUICK REFERENCE\n")
    f.write("="*70 + "\n\n")
    f.write(systems_df[['redshift', 'BH1_id', 'BH2_id', 'BH3_id', 
                         'BH1_halo_pynbody', 'BH2_halo_pynbody', 'BH3_halo_pynbody',
                         'num_unique_halos']].to_string(index=False))
    f.write("\n\n")
    f.write("Column definitions:\n")
    f.write("  BH1/2/3_id: Black hole particle ID (iord) - unique identifier\n")
    f.write("  BH1/2/3_halo_pynbody: Pynbody/AHF halo ID\n")
    f.write("  num_unique_halos: Number of distinct halos (1, 2, or 3)\n")

print(f"\nReference table saved to: {ref_table_path}")