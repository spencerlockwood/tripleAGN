import numpy as np
import pandas as pd
import glob
import os
from itertools import combinations

# Define paths
catalog_path = "/scratch/stlock/tripleAGNs/catalogs/1e40lum/catalogue_50kpc_1e40lum/"
output_dir = "/home/stlock/tripleAGN/analysis/"
os.makedirs(output_dir, exist_ok=True)

# Separation threshold used in detection (adjust to match your run)
SEPARATION_THRESHOLD = 50.0  # kpc

print("="*70)
print("SEARCHING FOR OVERLAPPING TRIPLE AGN SYSTEMS")
print("="*70)
print(f"Separation threshold: {SEPARATION_THRESHOLD} kpc\n")

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

# Storage for overlap cases
all_overlaps = []
snapshot_summary = []

# Read each file and check for overlaps
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
        
        print(f"Checking z={z:.2f} ({len(df)} triple system(s))...")
        
        if len(df) < 2:
            print(f"  Only 1 system - no overlaps possible")
            snapshot_summary.append({
                'redshift': z,
                'num_systems': len(df),
                'overlaps_found': 0,
                'overlapping_systems': 0
            })
            continue
        
        # Extract all BH positions for each system
        systems = []
        for idx, row in df.iterrows():
            system = {
                'system_id': idx,
                'BH_ids': [row['BH_id'], row['Neighbour1BH_id'], row['Neighbour2BH_id']],
                'BH_positions': [
                    np.array([row['BH_x'], row['BH_y'], row['BH_z']]),
                    np.array([row['Neighbour1BH_x'], row['Neighbour1BH_y'], row['Neighbour1BH_z']]),
                    np.array([row['Neighbour2BH_x'], row['Neighbour2BH_y'], row['Neighbour2BH_z']])
                ]
            }
            systems.append(system)
        
        # Check all pairs of systems for overlaps
        overlaps_in_snapshot = []
        overlapping_system_ids = set()
        
        for sys1, sys2 in combinations(range(len(systems)), 2):
            system1 = systems[sys1]
            system2 = systems[sys2]
            
            # Check if any BH from system1 is within separation threshold of any BH from system2
            overlaps = []
            for i, (bh1_id, bh1_pos) in enumerate(zip(system1['BH_ids'], system1['BH_positions'])):
                for j, (bh2_id, bh2_pos) in enumerate(zip(system2['BH_ids'], system2['BH_positions'])):
                    # Skip if it's the same BH (shouldn't happen but safety check)
                    if bh1_id == bh2_id:
                        continue
                    
                    # Calculate distance
                    distance = np.linalg.norm(bh1_pos - bh2_pos)
                    
                    # Check if within separation threshold
                    if distance < SEPARATION_THRESHOLD:
                        overlaps.append({
                            'system1_id': system1['system_id'],
                            'system2_id': system2['system_id'],
                            'system1_bh_index': i + 1,  # 1, 2, or 3
                            'system2_bh_index': j + 1,
                            'system1_bh_id': bh1_id,
                            'system2_bh_id': bh2_id,
                            'distance_kpc': distance
                        })
                        overlapping_system_ids.add(system1['system_id'])
                        overlapping_system_ids.add(system2['system_id'])
            
            if overlaps:
                overlaps_in_snapshot.extend(overlaps)
        
        # Report findings for this snapshot
        if overlaps_in_snapshot:
            print(f"  ⚠ OVERLAPS FOUND: {len(overlaps_in_snapshot)} BH pair(s) within threshold")
            print(f"  Systems involved: {len(overlapping_system_ids)} out of {len(df)}")
            
            for overlap in overlaps_in_snapshot:
                print(f"    System {overlap['system1_id']} (BH{overlap['system1_bh_index']}) <-> " +
                      f"System {overlap['system2_id']} (BH{overlap['system2_bh_index']}): " +
                      f"{overlap['distance_kpc']:.2f} kpc")
                
                # Add redshift to overlap record
                overlap['redshift'] = z
                all_overlaps.append(overlap)
        else:
            print(f"  ✓ No overlaps found")
        
        snapshot_summary.append({
            'redshift': z,
            'num_systems': len(df),
            'overlaps_found': len(overlaps_in_snapshot),
            'overlapping_systems': len(overlapping_system_ids)
        })
        
    except Exception as e:
        print(f"ERROR reading {os.path.basename(file_path)}: {e}")
        continue

# Create summary
summary_df = pd.DataFrame(snapshot_summary)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if len(all_overlaps) > 0:
    overlaps_df = pd.DataFrame(all_overlaps)
    
    print(f"\n⚠ OVERLAPS DETECTED!")
    print(f"Total BH pairs within threshold: {len(overlaps_df)}")
    print(f"Snapshots with overlaps: {(summary_df['overlaps_found'] > 0).sum()}")
    print(f"Total systems involved in overlaps: {summary_df['overlapping_systems'].sum()}")
    
    print(f"\nBreakdown by redshift:")
    for _, row in summary_df[summary_df['overlaps_found'] > 0].iterrows():
        print(f"  z={row['redshift']:.2f}: {row['overlaps_found']} overlap(s), " +
              f"{row['overlapping_systems']}/{row['num_systems']} systems affected")
    
    print(f"\nDistance distribution of overlapping pairs:")
    print(f"  Min: {overlaps_df['distance_kpc'].min():.2f} kpc")
    print(f"  Mean: {overlaps_df['distance_kpc'].mean():.2f} kpc")
    print(f"  Median: {overlaps_df['distance_kpc'].median():.2f} kpc")
    print(f"  Max: {overlaps_df['distance_kpc'].max():.2f} kpc")
    
    # Physical interpretation
    print("\n" + "="*70)
    print("PHYSICAL INTERPRETATION")
    print("="*70)
    print("These overlaps indicate:")
    print("  1. DENSE ENVIRONMENTS: Multiple triple systems in close proximity")
    print("  2. POTENTIAL LARGER SYSTEMS: Could be parts of quadruple/quintuple AGN systems")
    print("  3. MERGER COMPLEXITY: Complex multi-galaxy interactions")
    print(f"\nIMPLICATION: The algorithm's constraint that each BH can only be in")
    print(f"one system may be artificially splitting larger multi-AGN configurations.")
    
    # Save detailed overlap data
    #overlap_path = os.path.join(output_dir, "overlapping_triple_systems.csv")
    #overlaps_df.to_csv(overlap_path, index=False)
    #print(f"\nDetailed overlap data saved to: {overlap_path}")
    
    # Create visualization if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Number of overlaps by redshift
        ax1 = axes[0, 0]
        overlap_counts = summary_df[summary_df['overlaps_found'] > 0]
        if len(overlap_counts) > 0:
            ax1.bar(overlap_counts['redshift'], overlap_counts['overlaps_found'], 
                   width=0.05, color='#E63946', alpha=0.7)
        ax1.set_xlabel('Redshift', fontweight='bold')
        ax1.set_ylabel('Number of Overlapping BH Pairs', fontweight='bold')
        ax1.set_title('Overlaps by Redshift', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distance distribution
        ax2 = axes[0, 1]
        ax2.hist(overlaps_df['distance_kpc'], bins=20, color='#4169E1', alpha=0.7, edgecolor='black')
        ax2.axvline(SEPARATION_THRESHOLD, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold ({SEPARATION_THRESHOLD} kpc)')
        ax2.set_xlabel('Distance (kpc)', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title('Distance Distribution of Overlapping Pairs', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Fraction of systems with overlaps
        ax3 = axes[1, 0]
        overlap_fraction = summary_df[summary_df['num_systems'] > 0].copy()
        overlap_fraction['fraction'] = overlap_fraction['overlapping_systems'] / overlap_fraction['num_systems']
        affected_snaps = overlap_fraction[overlap_fraction['fraction'] > 0]
        if len(affected_snaps) > 0:
            ax3.scatter(affected_snaps['redshift'], affected_snaps['fraction'] * 100, 
                       s=80, alpha=0.7, color='#2A9D8F')
        ax3.set_xlabel('Redshift', fontweight='bold')
        ax3.set_ylabel('% of Systems with Overlaps', fontweight='bold')
        ax3.set_title('Fraction of Systems Affected by Overlaps', fontweight='bold')
        ax3.set_ylim(0, 105)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Which BH indices are overlapping (BH1, BH2, or BH3)
        ax4 = axes[1, 1]
        bh_indices = []
        for _, row in overlaps_df.iterrows():
            bh_indices.append(f"BH{int(row['system1_bh_index'])}")
            bh_indices.append(f"BH{int(row['system2_bh_index'])}")
        from collections import Counter
        bh_counts = Counter(bh_indices)
        ax4.bar(bh_counts.keys(), bh_counts.values(), color=['#E63946', '#4169E1', '#2A9D8F'])
        ax4.set_xlabel('BH Position in Triple System', fontweight='bold')
        ax4.set_ylabel('Count in Overlaps', fontweight='bold')
        ax4.set_title('Which BHs are Involved in Overlaps', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "overlap_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {plot_path}")
        
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
else:
    print(f"\n✓ NO OVERLAPS FOUND")
    print(f"All triple AGN systems are completely independent.")
    print(f"No BH from one system is within {SEPARATION_THRESHOLD} kpc of any BH from another system.")

# Save summary
summary_path = os.path.join(output_dir, "overlap_summary.txt")
with open(summary_path, 'w') as f:
    f.write("TRIPLE AGN SYSTEM OVERLAP ANALYSIS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Separation threshold used: {SEPARATION_THRESHOLD} kpc\n")
    f.write(f"Total snapshots analyzed: {len(summary_df)}\n")
    f.write(f"Total triple systems: {summary_df['num_systems'].sum()}\n\n")
    
    if len(all_overlaps) > 0:
        f.write("OVERLAPS DETECTED:\n")
        f.write(f"  Total overlapping BH pairs: {len(all_overlaps)}\n")
        f.write(f"  Snapshots with overlaps: {(summary_df['overlaps_found'] > 0).sum()}\n")
        f.write(f"  Systems involved: {summary_df['overlapping_systems'].sum()}\n\n")
        
        f.write("Detailed breakdown:\n")
        for _, row in summary_df[summary_df['overlaps_found'] > 0].iterrows():
            f.write(f"\n  z={row['redshift']:.2f}:\n")
            f.write(f"    Triple systems: {row['num_systems']}\n")
            f.write(f"    Overlapping pairs: {row['overlaps_found']}\n")
            f.write(f"    Systems affected: {row['overlapping_systems']}\n")
            
            # Get specific overlaps for this redshift
            z_overlaps = [o for o in all_overlaps if o['redshift'] == row['redshift']]
            for overlap in z_overlaps:
                f.write(f"      System {overlap['system1_id']} (BH{overlap['system1_bh_index']}, " +
                       f"ID={overlap['system1_bh_id']:.0f}) <-> " +
                       f"System {overlap['system2_id']} (BH{overlap['system2_bh_index']}, " +
                       f"ID={overlap['system2_bh_id']:.0f}): {overlap['distance_kpc']:.2f} kpc\n")
    else:
        f.write("NO OVERLAPS DETECTED\n")
        f.write("All triple AGN systems are completely independent.\n")

print(f"\nSummary saved to: {summary_path}")

# Save snapshot summary
#summary_csv_path = os.path.join(output_dir, "snapshot_overlap_summary_100.csv")
#summary_df.to_csv(summary_csv_path, index=False)
#print(f"Snapshot summary saved to: {summary_csv_path}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)