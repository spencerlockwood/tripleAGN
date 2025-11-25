import numpy as np
import pandas as pd
import glob
import os

print("="*70)
print("DIAGNOSING CATALOG INCONSISTENCY")
print("="*70)

# Define paths for both catalogs
catalog_50kpc = "/scratch/stlock/tripleAGNs/catalogs/1e40lum/catalogue_50kpc_1e40lum/"
catalog_100kpc = "/scratch/stlock/tripleAGNs/catalogs/1e40lum/catalogue_100kpc_1e40lum/"

output_dir = "/scratch/stlock/tripleAGNs/diagnosis/"
os.makedirs(output_dir, exist_ok=True)

def read_catalog(catalog_path, label):
    """Read all catalogs and return dict of {redshift: dataframe}"""
    
    print(f"\nReading {label} catalog...")
    
    files_pkl = sorted(glob.glob(f"{catalog_path}TripleAGN-Catalog*.pkl"))
    files_csv = sorted(glob.glob(f"{catalog_path}TripleAGN-Catalog*.csv"))
    files_h5 = sorted(glob.glob(f"{catalog_path}TripleAGN-Catalog*.h5"))
    
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
        print(f"  No files found!")
        return {}
    
    catalogs = {}
    
    for file_path in files_to_read:
        try:
            if file_format == 'pkl':
                df = pd.read_pickle(file_path)
            elif file_format == 'csv':
                df = pd.read_csv(file_path)
            elif file_format == 'h5':
                df = pd.read_hdf(file_path, key='tripleagn')
            
            filename = os.path.basename(file_path)
            if '-z' in filename:
                z_str = filename.split('-z')[1].split('.')[0] + '.' + filename.split('-z')[1].split('.')[1]
            else:
                z_str = filename.split('_z')[1].split('.')[0] + '.' + filename.split('_z')[1].split('.')[1]
            z = float(z_str)
            
            catalogs[z] = df
            
        except Exception as e:
            print(f"  Error reading {os.path.basename(file_path)}: {e}")
    
    print(f"  Loaded {len(catalogs)} snapshots")
    return catalogs

# Read both catalogs
cat_50 = read_catalog(catalog_50kpc, "50 kpc")
cat_100 = read_catalog(catalog_100kpc, "100 kpc")

# Find common redshifts
common_z = sorted(set(cat_50.keys()) & set(cat_100.keys()))
print(f"\nCommon redshifts: {len(common_z)}")

if len(common_z) == 0:
    print("ERROR: No common redshifts found!")
    exit(1)

# Analysis
print("\n" + "="*70)
print("COMPARING CATALOGS")
print("="*70)

inconsistencies = []

for z in common_z:
    df_50 = cat_50[z]
    df_100 = cat_100[z]
    
    # Skip if either is empty
    if len(df_50) == 0 and len(df_100) == 0:
        continue
    
    print(f"\nz = {z:.2f}:")
    print(f"  50 kpc:  {len(df_50)} systems")
    print(f"  100 kpc: {len(df_100)} systems")
    
    # Extract all BH IDs from each catalog
    bhs_50 = set()
    if len(df_50) > 0:
        bhs_50.update(df_50['BH_id'].values)
        bhs_50.update(df_50['Neighbour1BH_id'].values)
        bhs_50.update(df_50['Neighbour2BH_id'].values)
    
    bhs_100 = set()
    if len(df_100) > 0:
        bhs_100.update(df_100['BH_id'].values)
        bhs_100.update(df_100['Neighbour1BH_id'].values)
        bhs_100.update(df_100['Neighbour2BH_id'].values)
    
    print(f"  Unique BHs in 50 kpc:  {len(bhs_50)}")
    print(f"  Unique BHs in 100 kpc: {len(bhs_100)}")
    
    # BHs that are in 50 kpc but NOT in 100 kpc (THIS SHOULD NOT HAPPEN!)
    in_50_not_100 = bhs_50 - bhs_100
    
    # BHs that are in 100 kpc but not in 50 kpc (EXPECTED)
    in_100_not_50 = bhs_100 - bhs_50
    
    if len(in_50_not_100) > 0:
        print(f"  ⚠ INCONSISTENCY: {len(in_50_not_100)} BHs in 50kpc catalog but NOT in 100kpc!")
        print(f"    BH IDs: {list(in_50_not_100)[:5]}" + ("..." if len(in_50_not_100) > 5 else ""))
        
        # Find which systems these BHs belong to in each catalog
        for bh_id in list(in_50_not_100)[:3]:  # Check first 3
            # Find in 50 kpc catalog
            in_50_primary = df_50[df_50['BH_id'] == bh_id]
            in_50_n1 = df_50[df_50['Neighbour1BH_id'] == bh_id]
            in_50_n2 = df_50[df_50['Neighbour2BH_id'] == bh_id]
            
            if len(in_50_primary) > 0:
                row = in_50_primary.iloc[0]
                print(f"\n    BH {bh_id:.0f} in 50kpc catalog (Primary):")
                print(f"      Neighbors: {row['Neighbour1BH_id']:.0f}, {row['Neighbour2BH_id']:.0f}")
                print(f"      Separations: {row['Separation_3D_1_kpc']:.2f}, {row['Separation_3D_2_kpc']:.2f} kpc")
            elif len(in_50_n1) > 0:
                row = in_50_n1.iloc[0]
                print(f"\n    BH {bh_id:.0f} in 50kpc catalog (Neighbor1 of {row['BH_id']:.0f}):")
                print(f"      System: {row['BH_id']:.0f}, {bh_id:.0f}, {row['Neighbour2BH_id']:.0f}")
            elif len(in_50_n2) > 0:
                row = in_50_n2.iloc[0]
                print(f"\n    BH {bh_id:.0f} in 50kpc catalog (Neighbor2 of {row['BH_id']:.0f}):")
                print(f"      System: {row['BH_id']:.0f}, {row['Neighbour1BH_id']:.0f}, {bh_id:.0f}")
        
        inconsistencies.append({
            'redshift': z,
            'in_50_not_100': len(in_50_not_100),
            'in_100_not_50': len(in_100_not_50),
            'systems_50': len(df_50),
            'systems_100': len(df_100)
        })
    
    elif len(df_100) < len(df_50):
        print(f"  ⚠ UNEXPECTED: 100kpc has FEWER systems than 50kpc!")
        inconsistencies.append({
            'redshift': z,
            'in_50_not_100': len(in_50_not_100),
            'in_100_not_50': len(in_100_not_50),
            'systems_50': len(df_50),
            'systems_100': len(df_100)
        })
    
    else:
        print(f"  ✓ Consistent (100kpc is superset)")

# Summary
print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

if len(inconsistencies) > 0:
    print(f"\n⚠ INCONSISTENCIES FOUND in {len(inconsistencies)} snapshot(s)!")
    print("\nThis indicates the catalog building algorithm has issues:")
    print("  1. ORDER DEPENDENCY: The order BHs are processed matters")
    print("  2. GREEDY ASSIGNMENT: Once a BH is assigned, it can't be reconsidered")
    print("  3. NON-MONOTONIC: Larger thresholds don't always capture smaller threshold results")
    
    print("\nRECOMMENDED FIXES:")
    print("  Option 1: Remove the 'assigned_bhs' constraint entirely")
    print("            → Allow BHs to be in multiple systems")
    print("            → Post-process to handle overlaps")
    print("  Option 2: Sort BHs by some criteria (e.g., luminosity) before processing")
    print("            → At least make results deterministic")
    print("  Option 3: Use a global optimization approach")
    print("            → Find best non-overlapping set of triples")
    
    # Save detailed inconsistency report
    incon_df = pd.DataFrame(inconsistencies)
    incon_path = os.path.join(output_dir, "catalog_inconsistencies.csv")
    incon_df.to_csv(incon_path, index=False)
    print(f"\nInconsistency data saved to: {incon_path}")
    
else:
    print("\n✓ NO INCONSISTENCIES FOUND")
    print("The 100 kpc catalog properly contains all systems from 50 kpc catalog.")
    print("The overlap difference must be due to different system formation patterns.")

# Additional check: Are there cases where the SAME BH appears in different systems?
print("\n" + "="*70)
print("CHECKING FOR REUSED BHs WITHIN SAME CATALOG")
print("="*70)

def check_reused_bhs(catalog_dict, label):
    print(f"\n{label}:")
    total_reused = 0
    
    for z, df in catalog_dict.items():
        if len(df) == 0:
            continue
        
        all_bhs = []
        for _, row in df.iterrows():
            all_bhs.extend([row['BH_id'], row['Neighbour1BH_id'], row['Neighbour2BH_id']])
        
        unique_bhs = set(all_bhs)
        
        if len(all_bhs) != len(unique_bhs):
            reused = len(all_bhs) - len(unique_bhs)
            print(f"  z={z:.2f}: {reused} BH(s) appear in multiple systems!")
            total_reused += reused
    
    if total_reused == 0:
        print(f"  ✓ No BHs reused (constraint working correctly)")
    
    return total_reused

reused_50 = check_reused_bhs(cat_50, "50 kpc catalog")
reused_100 = check_reused_bhs(cat_100, "100 kpc catalog")

# Save full diagnostic report
report_path = os.path.join(output_dir, "diagnostic_report.txt")
with open(report_path, 'w') as f:
    f.write("CATALOG CONSISTENCY DIAGNOSTIC REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write("CATALOGS COMPARED:\n")
    f.write(f"  50 kpc catalog:  {len(cat_50)} snapshots\n")
    f.write(f"  100 kpc catalog: {len(cat_100)} snapshots\n")
    f.write(f"  Common redshifts: {len(common_z)}\n\n")
    
    if len(inconsistencies) > 0:
        f.write("INCONSISTENCIES DETECTED:\n")
        f.write(f"  Snapshots with issues: {len(inconsistencies)}\n\n")
        
        for incon in inconsistencies:
            f.write(f"\n  z={incon['redshift']:.2f}:\n")
            f.write(f"    Systems in 50kpc: {incon['systems_50']}\n")
            f.write(f"    Systems in 100kpc: {incon['systems_100']}\n")
            f.write(f"    BHs in 50 but not 100: {incon['in_50_not_100']}\n")
            f.write(f"    BHs in 100 but not 50: {incon['in_100_not_50']}\n")
        
        f.write("\n\nROOT CAUSE:\n")
        f.write("The 'assigned_bhs' constraint in find_triple_agn_systems() creates\n")
        f.write("an order-dependent algorithm. BHs are assigned to the first triple\n")
        f.write("system they're found in, preventing them from appearing in other\n")
        f.write("potentially valid systems.\n\n")
        
        f.write("With different separation thresholds, the FIRST system a BH gets\n")
        f.write("assigned to can be different, leading to completely different\n")
        f.write("final catalogs.\n")
        
    else:
        f.write("NO INCONSISTENCIES DETECTED\n")
        f.write("Catalogs appear consistent.\n")

print(f"\nFull diagnostic report saved to: {report_path}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)