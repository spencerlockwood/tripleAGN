import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Define paths
catalog_path = "/scratch/stlock/tripleAGNs/catalogs/1e43lum/catalogue_30kpc_1e43lum/"
output_dir = "/home/stlock/tripleAGN/analysis/"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("ANALYZING TRIPLE AGN HALO CONFIGURATIONS")
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

# Storage for all systems
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
        if '-z' in filename:
            z_str = filename.split('-z')[1].split('.')[0] + '.' + filename.split('-z')[1].split('.')[1]
        else:
            z_str = filename.split('_z')[1].split('.')[0] + '.' + filename.split('_z')[1].split('.')[1]
        z = float(z_str)
        
        # Extract detailed information for each system
        for idx, row in df.iterrows():
            # Calculate distances between BHs
            dx12 = row['BH_x'] - row['Neighbour1BH_x']
            dy12 = row['BH_y'] - row['Neighbour1BH_y']
            dz12 = row['BH_z'] - row['Neighbour1BH_z']
            sep_12 = np.sqrt(dx12**2 + dy12**2 + dz12**2)
            
            dx13 = row['BH_x'] - row['Neighbour2BH_x']
            dy13 = row['BH_y'] - row['Neighbour2BH_y']
            dz13 = row['BH_z'] - row['Neighbour2BH_z']
            sep_13 = np.sqrt(dx13**2 + dy13**2 + dz13**2)
            
            dx23 = row['Neighbour1BH_x'] - row['Neighbour2BH_x']
            dy23 = row['Neighbour1BH_y'] - row['Neighbour2BH_y']
            dz23 = row['Neighbour1BH_z'] - row['Neighbour2BH_z']
            sep_23 = np.sqrt(dx23**2 + dy23**2 + dz23**2)
            
            # Determine halo configuration
            halo1 = row['pynbody_haloid']
            halo2 = row['Neighbour1BH_pynbody_haloid']
            halo3 = row['Neighbour2BH_pynbody_haloid']
            unique_halos = len(set([halo1, halo2, halo3]))
            
            # Classify configuration
            if unique_halos == 1:
                config_type = "All in same halo"
            elif unique_halos == 2:
                config_type = "Two halos"
            else:
                config_type = "Three halos"
            
            system_info = {
                'redshift': z,
                'BH1_id': row['BH_id'],
                'BH2_id': row['Neighbour1BH_id'],
                'BH3_id': row['Neighbour2BH_id'],
                'BH1_halo': halo1,
                'BH2_halo': halo2,
                'BH3_halo': halo3,
                'num_unique_halos': unique_halos,
                'config_type': config_type,
                
                # BH properties
                'BH1_mass': row['BH_mass'],
                'BH2_mass': row['Neighbour1BH_mass'],
                'BH3_mass': row['Neighbour2BH_mass'],
                'BH1_Lbol': row['BH_Lbol'],
                'BH2_Lbol': row['Neighbour1BH_Lbol'],
                'BH3_Lbol': row['Neighbour2BH_Lbol'],
                
                # Separations
                'sep_BH1_BH2_kpc': sep_12,
                'sep_BH1_BH3_kpc': sep_13,
                'sep_BH2_BH3_kpc': sep_23,
                'min_separation': min(sep_12, sep_13, sep_23),
                'max_separation': max(sep_12, sep_13, sep_23),
                'mean_separation': np.mean([sep_12, sep_13, sep_23]),
                
                # Halo properties (if available)
                'Halo1_Mvir': row.get('Halo_Mvir', np.nan),
                'Halo1_M200': row.get('Halo_M200', np.nan),
                'Halo1_Rvir': row.get('Halo_Rvir', np.nan),
                'Halo1_R200': row.get('Halo_R200', np.nan),
            }
            
            all_systems.append(system_info)
        
    except Exception as e:
        print(f"Error reading {os.path.basename(file_path)}: {e}")
        continue

if len(all_systems) == 0:
    print("No triple AGN systems found!")
    exit(0)

# Convert to DataFrame
systems_df = pd.DataFrame(all_systems)

# Print summary statistics
print("\n" + "="*70)
print("HALO CONFIGURATION STATISTICS")
print("="*70)
print(f"Total triple AGN systems: {len(systems_df)}")
print(f"\nConfiguration breakdown:")
print(f"  All 3 BHs in same halo:  {(systems_df['num_unique_halos'] == 1).sum()} ({100*(systems_df['num_unique_halos'] == 1).sum()/len(systems_df):.1f}%)")
print(f"  BHs in 2 different halos: {(systems_df['num_unique_halos'] == 2).sum()} ({100*(systems_df['num_unique_halos'] == 2).sum()/len(systems_df):.1f}%)")
print(f"  BHs in 3 different halos: {(systems_df['num_unique_halos'] == 3).sum()} ({100*(systems_df['num_unique_halos'] == 3).sum()/len(systems_df):.1f}%)")

# Analyze systems in same halo
same_halo_systems = systems_df[systems_df['num_unique_halos'] == 1]
if len(same_halo_systems) > 0:
    print("\n" + "="*70)
    print("SYSTEMS WITH ALL BHs IN SAME HALO")
    print("="*70)
    print(f"Number of systems: {len(same_halo_systems)}")
    print(f"\nSeparation statistics:")
    print(f"  Min separation: {same_halo_systems['min_separation'].min():.2f} - {same_halo_systems['min_separation'].max():.2f} kpc")
    print(f"  Mean separation: {same_halo_systems['mean_separation'].mean():.2f} ± {same_halo_systems['mean_separation'].std():.2f} kpc")
    print(f"  Max separation: {same_halo_systems['max_separation'].min():.2f} - {same_halo_systems['max_separation'].max():.2f} kpc")
    
    print(f"\nBH mass statistics:")
    print(f"  BH1 mass: {same_halo_systems['BH1_mass'].median():.2e} Msun (median)")
    print(f"  BH2 mass: {same_halo_systems['BH2_mass'].median():.2e} Msun (median)")
    print(f"  BH3 mass: {same_halo_systems['BH3_mass'].median():.2e} Msun (median)")
    
    if not same_halo_systems['Halo1_Mvir'].isna().all():
        print(f"\nHost halo mass:")
        print(f"  Mvir: {same_halo_systems['Halo1_Mvir'].median():.2e} Msun (median)")
        print(f"  M200: {same_halo_systems['Halo1_M200'].median():.2e} Msun (median)")
        print(f"  Rvir: {same_halo_systems['Halo1_Rvir'].median():.2f} kpc (median)")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Configuration type distribution
ax1 = axes[0, 0]
config_counts = systems_df['config_type'].value_counts()
colors_pie = ['#E63946', '#4169E1', '#2A9D8F']
ax1.pie(config_counts.values, labels=config_counts.index, autopct='%1.1f%%',
        startangle=90, colors=colors_pie[:len(config_counts)])
ax1.set_title('Halo Configuration Distribution', fontweight='bold', fontsize=12)

# Plot 2: Separation vs halo configuration
ax2 = axes[0, 1]
for config in systems_df['config_type'].unique():
    subset = systems_df[systems_df['config_type'] == config]
    ax2.scatter(subset['redshift'], subset['mean_separation'], 
               label=config, alpha=0.6, s=60)
ax2.set_xlabel('Redshift', fontweight='bold')
ax2.set_ylabel('Mean BH Separation (kpc)', fontweight='bold')
ax2.set_title('BH Separation vs Redshift by Configuration', fontweight='bold', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Separation distribution by configuration
ax3 = axes[1, 0]
for config in systems_df['config_type'].unique():
    subset = systems_df[systems_df['config_type'] == config]
    ax3.hist(subset['mean_separation'], bins=15, alpha=0.6, label=config)
ax3.set_xlabel('Mean BH Separation (kpc)', fontweight='bold')
ax3.set_ylabel('Count', fontweight='bold')
ax3.set_title('Separation Distribution by Configuration', fontweight='bold', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Halo mass vs separation (for same-halo systems)
ax4 = axes[1, 1]
if not same_halo_systems['Halo1_Mvir'].isna().all():
    valid = same_halo_systems[same_halo_systems['Halo1_Mvir'].notna()]
    scatter = ax4.scatter(valid['Halo1_Mvir'], valid['mean_separation'],
                         c=valid['redshift'], cmap='viridis', s=60, alpha=0.7)
    ax4.set_xlabel('Halo Mvir (Msun)', fontweight='bold')
    ax4.set_ylabel('Mean BH Separation (kpc)', fontweight='bold')
    ax4.set_title('Halo Mass vs BH Separation\n(Same-halo systems)', fontweight='bold', fontsize=12)
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Redshift')
else:
    ax4.text(0.5, 0.5, 'No halo mass data available', 
            ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Halo Mass vs BH Separation', fontweight='bold', fontsize=12)

plt.tight_layout()

# Save plot
plot_path = os.path.join(output_dir, "halo_configuration_analysis.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {plot_path}")

# Save detailed analysis
analysis_path = os.path.join(output_dir, "halo_configuration_analysis.txt")
with open(analysis_path, 'w') as f:
    f.write("TRIPLE AGN HALO CONFIGURATION ANALYSIS\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Total systems: {len(systems_df)}\n\n")
    
    f.write("Configuration breakdown:\n")
    for config in systems_df['config_type'].unique():
        count = (systems_df['config_type'] == config).sum()
        pct = 100 * count / len(systems_df)
        f.write(f"  {config}: {count} ({pct:.1f}%)\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("SYSTEMS WITH ALL BHs IN SAME HALO\n")
    f.write("="*70 + "\n\n")
    
    if len(same_halo_systems) > 0:
        f.write(f"Count: {len(same_halo_systems)}\n\n")
        
        f.write("Possible physical interpretations:\n")
        f.write("  1. Post-merger systems: Multiple galaxies merged, BHs still separating\n")
        f.write("  2. Late-stage mergers: Halo finder sees single halo, BHs not yet merged\n")
        f.write("  3. Satellite galaxies: Small galaxies within larger host halo\n\n")
        
        f.write("Separation statistics:\n")
        f.write(f"  Mean: {same_halo_systems['mean_separation'].mean():.2f} ± {same_halo_systems['mean_separation'].std():.2f} kpc\n")
        f.write(f"  Median: {same_halo_systems['mean_separation'].median():.2f} kpc\n")
        f.write(f"  Range: {same_halo_systems['mean_separation'].min():.2f} - {same_halo_systems['mean_separation'].max():.2f} kpc\n\n")
        
        # Check if separations are comparable to virial radius
        if not same_halo_systems['Halo1_Rvir'].isna().all():
            valid_rvir = same_halo_systems[same_halo_systems['Halo1_Rvir'].notna()]
            sep_to_rvir = valid_rvir['mean_separation'] / valid_rvir['Halo1_Rvir']
            f.write(f"Separation relative to Rvir:\n")
            f.write(f"  Mean sep/Rvir: {sep_to_rvir.mean():.3f}\n")
            f.write(f"  Median sep/Rvir: {sep_to_rvir.median():.3f}\n")
            f.write(f"  => BH separations are typically {sep_to_rvir.median():.1%} of virial radius\n\n")
        
        f.write("Example systems:\n")
        for i, (idx, row) in enumerate(same_halo_systems.head(5).iterrows()):
            f.write(f"\n  System {i+1} (z={row['redshift']:.2f}):\n")
            f.write(f"    Halo ID: {row['BH1_halo']:.0f}\n")
            f.write(f"    BH IDs: {row['BH1_id']:.0f}, {row['BH2_id']:.0f}, {row['BH3_id']:.0f}\n")
            f.write(f"    Separations: {row['sep_BH1_BH2_kpc']:.2f}, {row['sep_BH1_BH3_kpc']:.2f}, {row['sep_BH2_BH3_kpc']:.2f} kpc\n")
            if not pd.isna(row['Halo1_Mvir']):
                f.write(f"    Halo mass: {row['Halo1_Mvir']:.2e} Msun\n")

print(f"Analysis saved to: {analysis_path}")

# Save data
data_path = os.path.join(output_dir, "halo_configuration_data.csv")
systems_df.to_csv(data_path, index=False)
print(f"Data saved to: {data_path}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)