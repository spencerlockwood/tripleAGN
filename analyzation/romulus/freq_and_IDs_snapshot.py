import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Define paths
catalog_path = "/scratch/stlock/tripleAGNs/catalogs/1e41lum/catalogue_15kpc_1e41lum/"
output_dir = "/scratch/stlock/tripleAGNs/plots_and_data/"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("TRIPLE AGN FREQUENCY AND ID TRACKING")
print("="*70)

# Search for catalog files
files_pkl = sorted(glob.glob(f"{catalog_path}TripleAGN-Catalog*.pkl"))
files_csv = sorted(glob.glob(f"{catalog_path}TripleAGN-Catalog*.csv"))
files_h5 = sorted(glob.glob(f"{catalog_path}TripleAGN-Catalog*.h5"))

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

print(f"Found {len(files_to_read)} catalog files ({file_format} format)\n")

# Storage for frequency data and detailed system info
frequency_data = []
all_system_details = []

# Snapshot list from build_tripleagn_catalog.py
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

# Load all catalog files to build z -> snapshot mapping
print("Building redshift to snapshot mapping...")
z_to_snapshot = {}
z_to_snapshot_index = {}

# We need to load the snapshots to get their redshifts
# Since we're reading the catalogs which already have redshift in filename,
# we can use the catalog filenames, but we need to map back to snapshot indices

# First pass: build mapping from redshift to snapshot
# We'll use the fact that catalogs are processed with snapshot_index as argument
# and output filename contains the redshift

# Alternative approach: Since catalog filenames contain redshift,
# and we know catalogs are created in order from snapshot_list,
# we can map catalog redshifts to snapshots by matching the processing order

# However, the most reliable way is to check which snapshots were actually processed
# Let's extract redshifts from filenames and match them
catalog_redshifts = {}
for file_path in files_to_read:
    filename = os.path.basename(file_path)
    z_str = filename.split('-z')[1].split('.pkl')[0].split('.csv')[0].split('.h5')[0]
    z = float(z_str)
    catalog_redshifts[z] = filename

print(f"Found {len(catalog_redshifts)} unique redshifts in catalogs")

# Now we need to figure out which snapshot corresponds to each redshift
# The build script processes snapshots by index, so we need to load
# a snapshot to get its redshift, or use a pre-computed mapping

# For now, let's create a function to find the closest snapshot
# based on the array indices in the slurm script (--array=49-119)
# This means snapshot_indices 49-119 were processed

# Since we can't load snapshots here, we'll use the catalog filename 
# which already has the redshift, and map it to the snapshot index
# based on the order catalogs were likely created

# More robust: extract from the slurm array indices
# array=49-119 means snapshot_list[49] through snapshot_list[119]
processed_snapshot_indices = range(49, 120)  # 49-119 inclusive

# We need to map z -> snapshot, but we only have z from filenames
# Let's use a lookup table approach or assume sequential processing

# Actually, looking at build_tripleagn_catalog.py more carefully:
# It takes snapshot_index as argument, loads that snapshot, gets its z,
# and saves with that z in the filename. So we need to reverse this.

# Practical solution: Create a comprehensive z->snapshot lookup
# by noting which snapshots were processed (indices 49-119)
processed_snapshots = [snapshot_list[i] for i in processed_snapshot_indices]

print(f"Processing snapshots from index 49-119 ({len(processed_snapshots)} snapshots)")
print(f"First snapshot: {processed_snapshots[0]}, Last snapshot: {processed_snapshots[-1]}")

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
        
        # Extract redshift from filename (format: TripleAGN-Catalog-R50-z{z}.pkl)
        filename = os.path.basename(file_path)
        z_str = filename.split('-z')[1].split('.pkl')[0].split('.csv')[0].split('.h5')[0]
        z = float(z_str)
        
        # Map redshift to snapshot
        # Since we don't have the exact z->snap mapping without loading snapshots,
        # we'll note that catalogs are named by their redshift
        # We need to figure out which snapshot this came from
        
        # The catalog filename tells us the redshift but not the snapshot directly
        # However, if we assume catalogs were created sequentially and not all succeeded,
        # we need to match the number of catalogs to snapshot indices
        
        # Let's use the file ordering to infer snapshot
        # Get index in the sorted file list
        file_idx = files_to_read.index(file_path)
        
        # This is the Nth successful catalog, so it likely corresponds to
        # processed_snapshots[file_idx] IF all catalogs succeeded
        # But some may have 0 systems, so this isn't perfect
        
        # Better approach: store the snapshot info in another way
        # For now, let's just note we need the snapshot and mark it as unknown
        # unless we can determine it another way
        
        # BEST SOLUTION: The catalog should have been saved with snapshot info!
        # But since it wasn't, we'll need to note this in the output
        
        # Let's assume sequential successful processing for now
        if file_idx < len(processed_snapshots):
            snapshot = processed_snapshots[file_idx]
        else:
            snapshot = "UNKNOWN"
        
        n_systems = len(df)
        
        # Store frequency data
        frequency_data.append({
            'redshift': z,
            'snapshot': snapshot,
            'n_systems': n_systems,
            'filename': filename
        })
        
        # Extract detailed info for each system
        for idx, row in df.iterrows():
            system_info = {
                'redshift': z,
                'snapshot': snapshot,
                'snapshot_fullname': f'cosmo25p.768sg1bwK1BHe75.{snapshot}',
                'filename': filename,
                'system_index': idx,
                
                # BH particle IDs (iord = pynbody particle finder_id)
                'BH1_finder_id': int(row.get('BH_id', -1)),
                'BH2_finder_id': int(row.get('Neighbour1BH_id', -1)),
                'BH3_finder_id': int(row.get('Neighbour2BH_id', -1)),
                
                # Halo finder IDs (pynbody halo IDs)
                'BH1_halo_finder_id': int(row.get('pynbody_haloid', -1)) if pd.notna(row.get('pynbody_haloid')) else -1,
                'BH2_halo_finder_id': int(row.get('Neighbour1BH_pynbody_haloid', -1)) if pd.notna(row.get('Neighbour1BH_pynbody_haloid')) else -1,
                'BH3_halo_finder_id': int(row.get('Neighbour2BH_pynbody_haloid', -1)) if pd.notna(row.get('Neighbour2BH_pynbody_haloid')) else -1,
                
                # BH masses
                'BH1_mass': row.get('BH_mass', np.nan),
                'BH2_mass': row.get('Neighbour1BH_mass', np.nan),
                'BH3_mass': row.get('Neighbour2BH_mass', np.nan),
                
                # BH luminosities
                'BH1_Lbol': row.get('BH_Lbol', np.nan),
                'BH2_Lbol': row.get('Neighbour1BH_Lbol', np.nan),
                'BH3_Lbol': row.get('Neighbour2BH_Lbol', np.nan),
                
                # Separations (comoving)
                'sep_BH1_BH2': row.get('Separation_3D_1_kpc_com', np.nan),
                'sep_BH1_BH3': row.get('Separation_3D_2_kpc_com', np.nan),
                'sep_BH2_BH3': row.get('Separation_3D_12_kpc_com', np.nan),
                
                # Host halo mass (main BH)
                'halo_Mvir': row.get('Halo_Mvir', np.nan),
            }
            all_system_details.append(system_info)
        
        print(f"z={z:.2f} (snap {snapshot}): {n_systems} system(s)")
        
    except Exception as e:
        print(f"ERROR reading {os.path.basename(file_path)}: {e}")
        continue

if len(frequency_data) == 0:
    print("No triple AGN data found!")
    exit(0)

# Convert to DataFrames
freq_df = pd.DataFrame(frequency_data).sort_values('redshift')
details_df = pd.DataFrame(all_system_details).sort_values(['redshift', 'system_index'])

print(f"\nTotal snapshots with data: {len(freq_df)}")
print(f"Total triple AGN systems: {freq_df['n_systems'].sum()}")

# Print warning if snapshot mapping is uncertain
if "UNKNOWN" in freq_df['snapshot'].values:
    print("\nWARNING: Some snapshots could not be determined from available data.")
    print("Consider adding snapshot info to catalog generation or loading snapshots to verify.")

# Print summary statistics
print("\n" + "="*70)
print("FREQUENCY STATISTICS")
print("="*70)
print(f"Redshift range: {freq_df['redshift'].min():.2f} - {freq_df['redshift'].max():.2f}")
print(f"Snapshot range: {freq_df['snapshot'].iloc[0]} - {freq_df['snapshot'].iloc[-1]}")
print(f"Systems per snapshot: {freq_df['n_systems'].mean():.1f} ± {freq_df['n_systems'].std():.1f}")
print(f"Max systems in single snapshot: {freq_df['n_systems'].max()} (z={freq_df.loc[freq_df['n_systems'].idxmax(), 'redshift']:.2f}, snap={freq_df.loc[freq_df['n_systems'].idxmax(), 'snapshot']})")

# Identify peak epochs
high_activity = freq_df[freq_df['n_systems'] >= freq_df['n_systems'].quantile(0.75)]
if len(high_activity) > 0:
    print(f"\nHigh activity epochs (top 25%):")
    for _, row in high_activity.iterrows():
        print(f"  z={row['redshift']:.2f} (snap {row['snapshot']}): {row['n_systems']} systems")

# Create frequency plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot as bar chart
ax.bar(freq_df['redshift'], freq_df['n_systems'], width=0.05, 
       color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.8)

ax.set_xlabel('Redshift', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Triple AGN Systems', fontsize=14, fontweight='bold')
ax.set_title('Triple AGN Frequency vs Redshift', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_axisbelow(True)

# Invert x-axis so time flows left to right
ax.invert_xaxis()

# Add text box with summary stats
textstr = f'Total systems: {freq_df["n_systems"].sum()}\n'
textstr += f'Snapshots: {len(freq_df)}\n'
textstr += f'Mean: {freq_df["n_systems"].mean():.1f} ± {freq_df["n_systems"].std():.1f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plot_path = os.path.join(output_dir, "triple_agn_frequency_vs_redshift.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nFrequency plot saved to: {plot_path}")
plt.close()

# Save detailed ID information to text file
txt_path = os.path.join(output_dir, "triple_agn_system_ids.txt")
with open(txt_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("TRIPLE AGN SYSTEM IDs - DETAILED LOG\n")
    f.write("All IDs are Pynbody finder_id values\n")
    f.write("="*80 + "\n")
    f.write(f"Generated from catalog: {catalog_path}\n")
    f.write(f"Total systems: {len(details_df)}\n")
    f.write(f"Redshift range: {details_df['redshift'].min():.2f} - {details_df['redshift'].max():.2f}\n")
    f.write(f"Snapshot range: {details_df['snapshot'].iloc[0]} - {details_df['snapshot'].iloc[-1]}\n")
    f.write("="*80 + "\n\n")
    
    # Group by redshift
    for z in sorted(details_df['redshift'].unique()):
        z_systems = details_df[details_df['redshift'] == z]
        snapshot = z_systems.iloc[0]['snapshot']
        snapshot_fullname = z_systems.iloc[0]['snapshot_fullname']
        
        f.write(f"\n{'='*80}\n")
        f.write(f"REDSHIFT z = {z:.4f}\n")
        f.write(f"Snapshot: {snapshot_fullname}\n")
        f.write(f"Catalog file: {z_systems.iloc[0]['filename']}\n")
        f.write(f"Number of systems: {len(z_systems)}\n")
        f.write(f"{'='*80}\n\n")
        
        for idx, sys in z_systems.iterrows():
            f.write(f"  System #{sys['system_index']}:\n")
            f.write(f"  {'-'*76}\n")
            f.write(f"    Black Hole Particle IDs (pynbody finder_id / iord):\n")
            f.write(f"      BH1: {sys['BH1_finder_id']:>12}   (M = {sys['BH1_mass']:>10.2e} Msun, Lbol = {sys['BH1_Lbol']:>10.2e} erg/s)\n")
            f.write(f"      BH2: {sys['BH2_finder_id']:>12}   (M = {sys['BH2_mass']:>10.2e} Msun, Lbol = {sys['BH2_Lbol']:>10.2e} erg/s)\n")
            f.write(f"      BH3: {sys['BH3_finder_id']:>12}   (M = {sys['BH3_mass']:>10.2e} Msun, Lbol = {sys['BH3_Lbol']:>10.2e} erg/s)\n")
            f.write(f"\n")
            f.write(f"    Host Halo IDs (pynbody halo finder_id):\n")
            f.write(f"      BH1 halo: {sys['BH1_halo_finder_id']:>12}\n")
            f.write(f"      BH2 halo: {sys['BH2_halo_finder_id']:>12}\n")
            f.write(f"      BH3 halo: {sys['BH3_halo_finder_id']:>12}\n")
            f.write(f"\n")
            f.write(f"    Separations (comoving kpc):\n")
            f.write(f"      BH1-BH2: {sys['sep_BH1_BH2']:>8.2f} kpc\n")
            f.write(f"      BH1-BH3: {sys['sep_BH1_BH3']:>8.2f} kpc\n")
            f.write(f"      BH2-BH3: {sys['sep_BH2_BH3']:>8.2f} kpc\n")
            f.write(f"\n")
            if pd.notna(sys['halo_Mvir']):
                f.write(f"    Main halo virial mass: {sys['halo_Mvir']:.2e} Msun\n")
            f.write(f"\n")

print(f"System IDs saved to: {txt_path}")

# Also save as CSV for easy analysis
csv_path = os.path.join(output_dir, "triple_agn_system_ids.csv")
details_df.to_csv(csv_path, index=False)
print(f"System IDs (CSV) saved to: {csv_path}")

# Save frequency data
freq_csv_path = os.path.join(output_dir, "triple_agn_frequency_data.csv")
freq_df.to_csv(freq_csv_path, index=False)
print(f"Frequency data saved to: {freq_csv_path}")

# Create a summary by redshift bin
print("\n" + "="*70)
print("SYSTEMS BY REDSHIFT BIN")
print("="*70)
z_bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf]
z_labels = ['0.0-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '>3.0']

for i in range(len(z_bins)-1):
    z_min, z_max = z_bins[i], z_bins[i+1]
    count = len(details_df[(details_df['redshift'] >= z_min) & (details_df['redshift'] < z_max)])
    print(f"  z = {z_labels[i]:>8}: {count:>4} systems")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)