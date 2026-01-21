#!/usr/bin/env python3
"""
Analyze triple AGN system configurations from build_tripleagn_catalog.py output.
Determines whether systems are compact three-body configurations or 
hierarchical (close pair + distant third) systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================
CATALOG_DIR = "/scratch/stlock/tripleAGNs/catalogs/1e41lum/ID_correct_catalogue_15kpc_1e41lum/"
OUTPUT_DIR = "/scratch/stlock/tripleAGNs/analysis/separation_plots/"
CATALOG_PATTERN = "TripleAGN-Catalog-R50-z*.pkl"

# ============================================================================
# LOAD DATA
# ============================================================================
def load_all_catalogs(catalog_dir, pattern):
    """Load all triple AGN catalogs and combine into single DataFrame."""
    catalog_files = sorted(glob.glob(str(Path(catalog_dir) / pattern)))
    
    if len(catalog_files) == 0:
        raise FileNotFoundError(f"No catalog files found matching {pattern} in {catalog_dir}")
    
    print(f"Found {len(catalog_files)} catalog files")
    
    all_data = []
    for fpath in catalog_files:
        try:
            df = pd.read_pickle(fpath)
            if len(df) > 0:
                all_data.append(df)
                print(f"  Loaded {len(df)} systems from {Path(fpath).name}")
        except Exception as e:
            print(f"  Error loading {fpath}: {e}")
    
    if len(all_data) == 0:
        raise ValueError("No valid data found in catalog files")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal triple AGN systems: {len(combined)}")
    return combined

# ============================================================================
# COMPUTE CONFIGURATION METRICS
# ============================================================================
def compute_configuration_metrics(df):
    """
    Compute metrics to classify triple system configurations.
    
    For each system with BH0 (main), BH1 (neighbor1), BH2 (neighbor2):
    - d01: separation between BH0 and BH1
    - d02: separation between BH0 and BH2  
    - d12: separation between BH1 and BH2
    """
    # Extract 3D separations
    d01 = df['Separation_3D_1_kpc_com'].values
    d02 = df['Separation_3D_2_kpc_com'].values
    d12 = df['Separation_3D_12_kpc_com'].values
    
    # For each triple, sort the three distances
    separations = np.column_stack([d01, d02, d12])
    sorted_seps = np.sort(separations, axis=1)
    
    d_min = sorted_seps[:, 0]  # closest pair
    d_mid = sorted_seps[:, 1]  # intermediate distance
    d_max = sorted_seps[:, 2]  # maximum distance
    
    # Configuration metrics
    compactness = d_max / d_min  # How hierarchical? >2-3 suggests hierarchy
    elongation = d_max / d_mid   # How elongated vs equilateral?
    
    # Classification
    # Hierarchical: one pair much closer than third (compactness > 2.5)
    # Compact: all three relatively similar distances (compactness < 2.0)
    is_hierarchical = compactness > 2.5
    is_compact = compactness < 2.0
    
    results = pd.DataFrame({
        'd_min': d_min,
        'd_mid': d_mid,
        'd_max': d_max,
        'd01': d01,
        'd02': d02,
        'd12': d12,
        'compactness': compactness,
        'elongation': elongation,
        'is_hierarchical': is_hierarchical,
        'is_compact': is_compact,
        'redshift': df['BH_redshift'].values if 'BH_redshift' in df.columns else np.nan
    })
    
    return results

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_separation_triangle(metrics, output_dir):
    """
    Triangle plot showing all three separations for each system.
    Each point represents one triple system.
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Color by configuration type
    colors = np.where(metrics['is_hierarchical'], 'red',
                     np.where(metrics['is_compact'], 'blue', 'orange'))
    
    scatter = ax.scatter(metrics['d01'], metrics['d02'], 
                        c=colors, s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Diagonal line (if d01 = d02)
    max_val = max(metrics['d01'].max(), metrics['d02'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='d01 = d02')
    
    ax.set_xlabel('Separation BH0-BH1 (comoving kpc)', fontsize=12)
    ax.set_ylabel('Separation BH0-BH2 (comoving kpc)', fontsize=12)
    ax.set_title('Triple AGN Separation Space', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Legend
    red_patch = mpatches.Patch(color='red', label=f'Hierarchical (N={metrics["is_hierarchical"].sum()})')
    blue_patch = mpatches.Patch(color='blue', label=f'Compact (N={metrics["is_compact"].sum()})')
    orange_patch = mpatches.Patch(color='orange', label=f'Intermediate (N={(~metrics["is_hierarchical"] & ~metrics["is_compact"]).sum()})')
    ax.legend(handles=[red_patch, blue_patch, orange_patch], fontsize=10)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'separation_triangle.png', dpi=300, bbox_inches='tight')
    print(f"Saved: separation_triangle.png")
    plt.close()

def plot_compactness_histogram(metrics, output_dir):
    """
    Histogram of compactness ratio (d_max / d_min).
    Shows distribution of hierarchical vs compact systems.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(metrics['compactness'], bins=30, color='steelblue', 
            edgecolor='black', alpha=0.7)
    
    # Mark classification thresholds
    ax.axvline(2.0, color='blue', linestyle='--', linewidth=2, 
               label='Compact threshold (< 2.0)')
    ax.axvline(2.5, color='red', linestyle='--', linewidth=2,
               label='Hierarchical threshold (> 2.5)')
    
    ax.set_xlabel('Compactness Ratio (d_max / d_min)', fontsize=12)
    ax.set_ylabel('Number of Systems', fontsize=12)
    ax.set_title('Distribution of Triple AGN Configurations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text box with statistics
    textstr = f'Total systems: {len(metrics)}\n'
    textstr += f'Hierarchical: {metrics["is_hierarchical"].sum()} ({100*metrics["is_hierarchical"].mean():.1f}%)\n'
    textstr += f'Compact: {metrics["is_compact"].sum()} ({100*metrics["is_compact"].mean():.1f}%)\n'
    textstr += f'Intermediate: {(~metrics["is_hierarchical"] & ~metrics["is_compact"]).sum()}'
    
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'compactness_histogram.png', dpi=300, bbox_inches='tight')
    print(f"Saved: compactness_histogram.png")
    plt.close()

def plot_separation_distributions(metrics, output_dir):
    """
    Compare distributions of minimum, middle, and maximum separations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Minimum separation
    axes[0].hist(metrics['d_min'], bins=20, color='green', 
                 edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Minimum Separation (comoving kpc)', fontsize=11)
    axes[0].set_ylabel('Number of Systems', fontsize=11)
    axes[0].set_title('Closest Pair Distance', fontsize=12, fontweight='bold')
    axes[0].axvline(metrics['d_min'].median(), color='darkgreen', 
                    linestyle='--', linewidth=2, label=f'Median: {metrics["d_min"].median():.2f} kpc')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Middle separation
    axes[1].hist(metrics['d_mid'], bins=20, color='orange', 
                 edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Middle Separation (comoving kpc)', fontsize=11)
    axes[1].set_ylabel('Number of Systems', fontsize=11)
    axes[1].set_title('Intermediate Distance', fontsize=12, fontweight='bold')
    axes[1].axvline(metrics['d_mid'].median(), color='darkorange', 
                    linestyle='--', linewidth=2, label=f'Median: {metrics["d_mid"].median():.2f} kpc')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Maximum separation
    axes[2].hist(metrics['d_max'], bins=20, color='red', 
                 edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Maximum Separation (comoving kpc)', fontsize=11)
    axes[2].set_ylabel('Number of Systems', fontsize=11)
    axes[2].set_title('Farthest Distance', fontsize=12, fontweight='bold')
    axes[2].axvline(metrics['d_max'].median(), color='darkred', 
                    linestyle='--', linewidth=2, label=f'Median: {metrics["d_max"].median():.2f} kpc')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'separation_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: separation_distributions.png")
    plt.close()

def plot_2d_vs_3d_separations(metrics, output_dir):
    """
    Compare 2D (projected) vs 3D separations to understand geometry.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Need to get 2D separations from original dataframe
    # This plot needs to be called with original df, so skip for now
    # or compute from the metrics we have
    
    ax.scatter(metrics['d_min'], metrics['d_max'], 
              c=metrics['compactness'], cmap='viridis', 
              s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Compactness Ratio', fontsize=11)
    
    ax.set_xlabel('Minimum Separation (comoving kpc)', fontsize=12)
    ax.set_ylabel('Maximum Separation (comoving kpc)', fontsize=12)
    ax.set_title('Min vs Max Separation in Triple Systems', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'min_vs_max_separation.png', dpi=300, bbox_inches='tight')
    print(f"Saved: min_vs_max_separation.png")
    plt.close()

def plot_shape_diagram(metrics, output_dir):
    """
    Scatter plot of compactness vs elongation to classify shapes.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = np.where(metrics['is_hierarchical'], 'red',
                     np.where(metrics['is_compact'], 'blue', 'orange'))
    
    ax.scatter(metrics['compactness'], metrics['elongation'],
              c=colors, s=60, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Add region labels
    ax.axvline(2.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(1.5, color='purple', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax.text(0.95, 0.95, 'Hierarchical\n(close pair + distant)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    ax.text(0.05, 0.05, 'Compact\nEquilateral', 
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    
    ax.set_xlabel('Compactness (d_max / d_min)', fontsize=12)
    ax.set_ylabel('Elongation (d_max / d_mid)', fontsize=12)
    ax.set_title('Triple AGN Shape Classification', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Legend
    red_patch = mpatches.Patch(color='red', label='Hierarchical')
    blue_patch = mpatches.Patch(color='blue', label='Compact')
    orange_patch = mpatches.Patch(color='orange', label='Intermediate')
    ax.legend(handles=[red_patch, blue_patch, orange_patch], fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'shape_classification.png', dpi=300, bbox_inches='tight')
    print(f"Saved: shape_classification.png")
    plt.close()

def plot_schematic_examples(metrics, output_dir):
    """
    Create schematic diagrams showing example configurations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Find representative examples
    hierarchical_idx = metrics[metrics['is_hierarchical']].index[0] if metrics['is_hierarchical'].any() else None
    compact_idx = metrics[metrics['is_compact']].index[0] if metrics['is_compact'].any() else None
    intermediate_idx = metrics[~metrics['is_hierarchical'] & ~metrics['is_compact']].index[0] if (~metrics['is_hierarchical'] & ~metrics['is_compact']).any() else None
    
    examples = [
        (hierarchical_idx, 'Hierarchical', 'red', 0),
        (compact_idx, 'Compact', 'blue', 1),
        (intermediate_idx, 'Intermediate', 'orange', 2)
    ]
    
    for idx, label, color, ax_idx in examples:
        ax = axes[ax_idx]
        if idx is None:
            ax.text(0.5, 0.5, f'No {label}\nexamples found', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            continue
        
        # Get separations for this system
        d01 = metrics.loc[idx, 'd01']
        d02 = metrics.loc[idx, 'd02']
        d12 = metrics.loc[idx, 'd12']
        
        # Place BH0 at origin
        bh0 = np.array([0, 0])
        # Place BH1 along x-axis
        bh1 = np.array([d01, 0])
        # Place BH2 using law of cosines
        cos_angle = (d01**2 + d02**2 - d12**2) / (2 * d01 * d02)
        cos_angle = np.clip(cos_angle, -1, 1)  # numerical safety
        angle = np.arccos(cos_angle)
        bh2 = np.array([d02 * np.cos(angle), d02 * np.sin(angle)])
        
        # Plot the triangle
        triangle = plt.Polygon([bh0, bh1, bh2], fill=False, 
                               edgecolor=color, linewidth=2)
        ax.add_patch(triangle)
        
        # Plot BH positions
        ax.scatter(*bh0, s=200, c=color, marker='o', edgecolors='black', linewidth=2, zorder=10)
        ax.scatter(*bh1, s=200, c=color, marker='o', edgecolors='black', linewidth=2, zorder=10)
        ax.scatter(*bh2, s=200, c=color, marker='o', edgecolors='black', linewidth=2, zorder=10)
        
        # Add labels
        ax.text(bh0[0], bh0[1]-0.7, 'BH0', ha='center', fontsize=10, fontweight='bold')
        ax.text(bh1[0], bh1[1]-0.7, 'BH1', ha='center', fontsize=10, fontweight='bold')
        ax.text(bh2[0], bh2[1]+0.7, 'BH2', ha='center', fontsize=10, fontweight='bold')
        
        # Add distance labels
        ax.text((bh0[0]+bh1[0])/2, (bh0[1]+bh1[1])/2-0.4, f'{d01:.1f}', 
               ha='center', fontsize=9, style='italic')
        ax.text((bh0[0]+bh2[0])/2-0.5, (bh0[1]+bh2[1])/2, f'{d02:.1f}', 
               ha='center', fontsize=9, style='italic')
        ax.text((bh1[0]+bh2[0])/2+0.5, (bh1[1]+bh2[1])/2, f'{d12:.1f}', 
               ha='center', fontsize=9, style='italic')
        
        ax.set_aspect('equal')
        ax.set_title(f'{label}\nCompactness: {metrics.loc[idx, "compactness"]:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Distance (comoving kpc)', fontsize=10)
        ax.set_ylabel('Distance (comoving kpc)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set limits with padding
        all_x = [bh0[0], bh1[0], bh2[0]]
        all_y = [bh0[1], bh1[1], bh2[1]]
        padding = 1.5
        ax.set_xlim(min(all_x)-padding, max(all_x)+padding)
        ax.set_ylim(min(all_y)-padding, max(all_y)+padding)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'schematic_examples.png', dpi=300, bbox_inches='tight')
    print(f"Saved: schematic_examples.png")
    plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
def print_summary_statistics(metrics):
    """Print comprehensive summary of triple AGN configurations."""
    print("\n" + "="*70)
    print("TRIPLE AGN CONFIGURATION ANALYSIS")
    print("="*70)
    
    print(f"\nTotal triple systems analyzed: {len(metrics)}")
    
    print("\n--- Configuration Types ---")
    n_hierarchical = metrics['is_hierarchical'].sum()
    n_compact = metrics['is_compact'].sum()
    n_intermediate = (~metrics['is_hierarchical'] & ~metrics['is_compact']).sum()
    
    print(f"Hierarchical (close pair + distant): {n_hierarchical} ({100*n_hierarchical/len(metrics):.1f}%)")
    print(f"Compact (three-body):                {n_compact} ({100*n_compact/len(metrics):.1f}%)")
    print(f"Intermediate:                        {n_intermediate} ({100*n_intermediate/len(metrics):.1f}%)")
    
    print("\n--- Separation Statistics (comoving kpc) ---")
    print(f"Minimum separation:")
    print(f"  Mean:   {metrics['d_min'].mean():.2f} ± {metrics['d_min'].std():.2f}")
    print(f"  Median: {metrics['d_min'].median():.2f}")
    print(f"  Range:  {metrics['d_min'].min():.2f} - {metrics['d_min'].max():.2f}")
    
    print(f"\nMiddle separation:")
    print(f"  Mean:   {metrics['d_mid'].mean():.2f} ± {metrics['d_mid'].std():.2f}")
    print(f"  Median: {metrics['d_mid'].median():.2f}")
    print(f"  Range:  {metrics['d_mid'].min():.2f} - {metrics['d_mid'].max():.2f}")
    
    print(f"\nMaximum separation:")
    print(f"  Mean:   {metrics['d_max'].mean():.2f} ± {metrics['d_max'].std():.2f}")
    print(f"  Median: {metrics['d_max'].median():.2f}")
    print(f"  Range:  {metrics['d_max'].min():.2f} - {metrics['d_max'].max():.2f}")
    
    print("\n--- Configuration Metrics ---")
    print(f"Compactness ratio (d_max/d_min):")
    print(f"  Mean:   {metrics['compactness'].mean():.2f} ± {metrics['compactness'].std():.2f}")
    print(f"  Median: {metrics['compactness'].median():.2f}")
    
    print(f"\nElongation ratio (d_max/d_mid):")
    print(f"  Mean:   {metrics['elongation'].mean():.2f} ± {metrics['elongation'].std():.2f}")
    print(f"  Median: {metrics['elongation'].median():.2f}")
    
    print("\n" + "="*70)
    
    # Answer the key questions
    print("\nKEY FINDINGS:")
    print("-" * 70)
    if n_hierarchical > n_compact:
        print("✓ Triples are PREDOMINANTLY hierarchical (close pair + distant third)")
    elif n_compact > n_hierarchical:
        print("✓ Triples are PREDOMINANTLY compact three-body systems")
    else:
        print("✓ Triples show MIXED configurations (no clear preference)")
    
    if metrics['d_min'].median() < 5:
        print(f"✓ Closest pairs are typically VERY CLOSE (median: {metrics['d_min'].median():.2f} kpc)")
    
    if metrics['compactness'].median() > 2.5:
        print(f"✓ Systems show STRONG HIERARCHY (median compactness: {metrics['compactness'].median():.2f})")
    elif metrics['compactness'].median() < 2.0:
        print(f"✓ Systems are RELATIVELY COMPACT (median compactness: {metrics['compactness'].median():.2f})")
    
    print("="*70 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main analysis pipeline."""
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("Loading triple AGN catalogs...")
    df = load_all_catalogs(CATALOG_DIR, CATALOG_PATTERN)
    
    print("\nComputing configuration metrics...")
    metrics = compute_configuration_metrics(df)
    
    print("\nGenerating plots...")
    plot_separation_triangle(metrics, OUTPUT_DIR)
    plot_compactness_histogram(metrics, OUTPUT_DIR)
    plot_separation_distributions(metrics, OUTPUT_DIR)
    plot_2d_vs_3d_separations(metrics, OUTPUT_DIR)
    plot_shape_diagram(metrics, OUTPUT_DIR)
    plot_schematic_examples(metrics, OUTPUT_DIR)
    
    print_summary_statistics(metrics)
    
    # Save metrics to file
    metrics_path = Path(OUTPUT_DIR) / 'triple_agn_metrics.csv'
    metrics.to_csv(metrics_path, index=False)
    print(f"\nSaved configuration metrics to: {metrics_path}")
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()