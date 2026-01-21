#!/usr/bin/env python3
"""
Analyze luminosity distributions and patterns in triple AGN systems.
Determines whether high/low luminosity AGNs cluster together and 
identifies common luminosity hierarchies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import glob
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================
CATALOG_DIR = "/scratch/stlock/tripleAGNs/catalogs/1e41lum/ID_correct_catalogue_15kpc_1e41lum/"
OUTPUT_DIR = "/scratch/stlock/tripleAGNs/analysis/luminosity_plots/"
CATALOG_PATTERN = "TripleAGN-Catalog-R50-z*.pkl"

# Luminosity thresholds for classification (erg/s)
LBOL_HIGH = 1e44      # High luminosity threshold
LBOL_MEDIUM = 1e43    # Medium luminosity threshold
# Below 1e43 is low (but all should be > 1e41 by selection)

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
# COMPUTE LUMINOSITY METRICS
# ============================================================================
def compute_luminosity_metrics(df):
    """
    Extract and analyze luminosity patterns in triple systems.
    
    For each system:
    - L0: luminosity of main BH
    - L1: luminosity of neighbor 1
    - L2: luminosity of neighbor 2
    """
    # Extract bolometric luminosities
    L0 = df['BH_Lbol'].values
    L1 = df['Neighbour1BH_Lbol'].values
    L2 = df['Neighbour2BH_Lbol'].values
    
    # For each triple, sort luminosities
    luminosities = np.column_stack([L0, L1, L2])
    sorted_lums = np.sort(luminosities, axis=1)[:, ::-1]  # Sort descending
    
    L_max = sorted_lums[:, 0]  # Brightest AGN
    L_mid = sorted_lums[:, 1]  # Middle AGN
    L_min = sorted_lums[:, 2]  # Faintest AGN
    
    # Compute luminosity ratios
    ratio_max_mid = L_max / L_mid
    ratio_max_min = L_max / L_min
    ratio_mid_min = L_mid / L_min
    
    # Luminosity hierarchy metric: are luminosities clustered or spread?
    lum_spread = np.log10(L_max / L_min)  # in dex
    
    # Classify each BH by luminosity
    def classify_lum(L):
        if L >= LBOL_HIGH:
            return 'High'
        elif L >= LBOL_MEDIUM:
            return 'Medium'
        else:
            return 'Low'
    
    class0 = np.array([classify_lum(l) for l in L0])
    class1 = np.array([classify_lum(l) for l in L1])
    class2 = np.array([classify_lum(l) for l in L2])
    
    # Count luminosity class composition for each system
    def count_classes(c0, c1, c2):
        classes = [c0, c1, c2]
        n_high = classes.count('High')
        n_medium = classes.count('Medium')
        n_low = classes.count('Low')
        return n_high, n_medium, n_low
    
    compositions = np.array([count_classes(c0, c1, c2) 
                            for c0, c1, c2 in zip(class0, class1, class2)])
    
    n_high = compositions[:, 0]
    n_medium = compositions[:, 1]
    n_low = compositions[:, 2]
    
    # Create composition labels
    comp_labels = []
    for nh, nm, nl in zip(n_high, n_medium, n_low):
        if nh == 3:
            comp_labels.append('3H-0M-0L')
        elif nh == 2 and nm == 1:
            comp_labels.append('2H-1M-0L')
        elif nh == 2 and nl == 1:
            comp_labels.append('2H-0M-1L')
        elif nh == 1 and nm == 2:
            comp_labels.append('1H-2M-0L')
        elif nh == 1 and nm == 1 and nl == 1:
            comp_labels.append('1H-1M-1L')
        elif nh == 1 and nl == 2:
            comp_labels.append('1H-0M-2L')
        elif nm == 3:
            comp_labels.append('0H-3M-0L')
        elif nm == 2 and nl == 1:
            comp_labels.append('0H-2M-1L')
        elif nm == 1 and nl == 2:
            comp_labels.append('0H-1M-2L')
        elif nl == 3:
            comp_labels.append('0H-0M-3L')
        else:
            comp_labels.append('Other')
    
    results = pd.DataFrame({
        'L0': L0,
        'L1': L1,
        'L2': L2,
        'L_max': L_max,
        'L_mid': L_mid,
        'L_min': L_min,
        'ratio_max_mid': ratio_max_mid,
        'ratio_max_min': ratio_max_min,
        'ratio_mid_min': ratio_mid_min,
        'lum_spread_dex': lum_spread,
        'class0': class0,
        'class1': class1,
        'class2': class2,
        'n_high': n_high,
        'n_medium': n_medium,
        'n_low': n_low,
        'composition': comp_labels
    })
    
    return results

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_luminosity_distributions(metrics, output_dir):
    """
    Plot distributions of luminosities for all three BHs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # All luminosities combined
    ax = axes[0, 0]
    all_lums = np.concatenate([metrics['L0'], metrics['L1'], metrics['L2']])
    ax.hist(np.log10(all_lums), bins=30, color='steelblue', 
            edgecolor='black', alpha=0.7)
    ax.axvline(np.log10(LBOL_HIGH), color='red', linestyle='--', 
               linewidth=2, label=f'High: {LBOL_HIGH:.0e} erg/s')
    ax.axvline(np.log10(LBOL_MEDIUM), color='orange', linestyle='--', 
               linewidth=2, label=f'Medium: {LBOL_MEDIUM:.0e} erg/s')
    ax.set_xlabel('log₁₀(Lbol) [erg/s]', fontsize=12)
    ax.set_ylabel('Number of AGN', fontsize=12)
    ax.set_title('All AGN Luminosities', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Sorted luminosities (max, mid, min)
    ax = axes[0, 1]
    ax.hist(np.log10(metrics['L_max']), bins=25, color='red', 
            alpha=0.5, edgecolor='black', label='Brightest')
    ax.hist(np.log10(metrics['L_mid']), bins=25, color='orange', 
            alpha=0.5, edgecolor='black', label='Middle')
    ax.hist(np.log10(metrics['L_min']), bins=25, color='blue', 
            alpha=0.5, edgecolor='black', label='Faintest')
    ax.set_xlabel('log₁₀(Lbol) [erg/s]', fontsize=12)
    ax.set_ylabel('Number of Systems', fontsize=12)
    ax.set_title('Luminosity by Rank in Triple', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Luminosity spread
    ax = axes[1, 0]
    ax.hist(metrics['lum_spread_dex'], bins=25, color='purple', 
            edgecolor='black', alpha=0.7)
    ax.axvline(metrics['lum_spread_dex'].median(), color='darkviolet', 
               linestyle='--', linewidth=2, 
               label=f'Median: {metrics["lum_spread_dex"].median():.2f} dex')
    ax.set_xlabel('Luminosity Spread (log₁₀(L_max/L_min)) [dex]', fontsize=12)
    ax.set_ylabel('Number of Systems', fontsize=12)
    ax.set_title('Luminosity Hierarchy Strength', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Luminosity ratios
    ax = axes[1, 1]
    ax.hist(np.log10(metrics['ratio_max_mid']), bins=25, color='green', 
            alpha=0.5, edgecolor='black', label='L_max / L_mid')
    ax.hist(np.log10(metrics['ratio_mid_min']), bins=25, color='cyan', 
            alpha=0.5, edgecolor='black', label='L_mid / L_min')
    ax.set_xlabel('log₁₀(Luminosity Ratio)', fontsize=12)
    ax.set_ylabel('Number of Systems', fontsize=12)
    ax.set_title('Pairwise Luminosity Ratios', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'luminosity_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: luminosity_distributions.png")
    plt.close()

def plot_composition_pie_chart(metrics, output_dir):
    """
    Pie chart showing the most common luminosity compositions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Count compositions
    comp_counts = metrics['composition'].value_counts()
    
    # Full pie chart
    ax = axes[0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(comp_counts)))
    wedges, texts, autotexts = ax.pie(comp_counts.values, labels=comp_counts.index,
                                        autopct='%1.1f%%', startangle=90,
                                        colors=colors, textprops={'fontsize': 10})
    ax.set_title('Luminosity Composition of Triple Systems', 
                fontsize=14, fontweight='bold')
    
    # Simplified groupings
    ax = axes[1]
    
    # Group by pattern
    one_bright_two_dim = metrics[(metrics['n_high'] == 1) & 
                                 ((metrics['n_medium'] == 2) | (metrics['n_low'] == 2))].shape[0]
    two_bright_one_dim = metrics[(metrics['n_high'] == 2) & 
                                 ((metrics['n_medium'] == 1) | (metrics['n_low'] == 1))].shape[0]
    all_similar = metrics[(metrics['n_high'] == 3) | (metrics['n_medium'] == 3) | 
                         (metrics['n_low'] == 3)].shape[0]
    mixed = len(metrics) - one_bright_two_dim - two_bright_one_dim - all_similar
    
    pattern_labels = ['1 Bright + 2 Dim', '2 Bright + 1 Dim', 'All Similar', 'Mixed']
    pattern_counts = [one_bright_two_dim, two_bright_one_dim, all_similar, mixed]
    pattern_colors = ['gold', 'tomato', 'lightblue', 'lightgreen']
    
    wedges, texts, autotexts = ax.pie(pattern_counts, labels=pattern_labels,
                                        autopct='%1.1f%%', startangle=90,
                                        colors=pattern_colors, textprops={'fontsize': 11})
    ax.set_title('Simplified Luminosity Patterns', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'composition_pie_charts.png', dpi=300, bbox_inches='tight')
    print(f"Saved: composition_pie_charts.png")
    plt.close()

def plot_luminosity_correlation_matrix(metrics, output_dir):
    """
    Scatter matrix showing correlations between L0, L1, L2.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # L0 vs L1
    ax = axes[0, 0]
    ax.scatter(np.log10(metrics['L0']), np.log10(metrics['L1']), 
              alpha=0.5, s=30, edgecolors='k', linewidth=0.3)
    ax.plot([41, 46], [41, 46], 'r--', alpha=0.5, linewidth=2, label='L0 = L1')
    ax.set_xlabel('log₁₀(L₀) [erg/s]', fontsize=11)
    ax.set_ylabel('log₁₀(L₁) [erg/s]', fontsize=11)
    ax.set_title('Main BH vs Neighbor 1', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Compute correlation
    corr01 = np.corrcoef(np.log10(metrics['L0']), np.log10(metrics['L1']))[0, 1]
    ax.text(0.05, 0.95, f'r = {corr01:.3f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # L0 vs L2
    ax = axes[0, 1]
    ax.scatter(np.log10(metrics['L0']), np.log10(metrics['L2']), 
              alpha=0.5, s=30, edgecolors='k', linewidth=0.3, color='orange')
    ax.plot([41, 46], [41, 46], 'r--', alpha=0.5, linewidth=2, label='L0 = L2')
    ax.set_xlabel('log₁₀(L₀) [erg/s]', fontsize=11)
    ax.set_ylabel('log₁₀(L₂) [erg/s]', fontsize=11)
    ax.set_title('Main BH vs Neighbor 2', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    corr02 = np.corrcoef(np.log10(metrics['L0']), np.log10(metrics['L2']))[0, 1]
    ax.text(0.05, 0.95, f'r = {corr02:.3f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # L1 vs L2
    ax = axes[1, 0]
    ax.scatter(np.log10(metrics['L1']), np.log10(metrics['L2']), 
              alpha=0.5, s=30, edgecolors='k', linewidth=0.3, color='green')
    ax.plot([41, 46], [41, 46], 'r--', alpha=0.5, linewidth=2, label='L1 = L2')
    ax.set_xlabel('log₁₀(L₁) [erg/s]', fontsize=11)
    ax.set_ylabel('log₁₀(L₂) [erg/s]', fontsize=11)
    ax.set_title('Neighbor 1 vs Neighbor 2', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    corr12 = np.corrcoef(np.log10(metrics['L1']), np.log10(metrics['L2']))[0, 1]
    ax.text(0.05, 0.95, f'r = {corr12:.3f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Correlation summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "Luminosity Correlations\n\n"
    summary_text += f"L₀ vs L₁: r = {corr01:.3f}\n"
    summary_text += f"L₀ vs L₂: r = {corr02:.3f}\n"
    summary_text += f"L₁ vs L₂: r = {corr12:.3f}\n\n"
    
    avg_corr = np.mean([corr01, corr02, corr12])
    summary_text += f"Average: r = {avg_corr:.3f}\n\n"
    
    if avg_corr > 0.5:
        summary_text += "→ Strong positive correlation\n"
        summary_text += "→ High-L AGN cluster together\n"
        summary_text += "→ Low-L AGN cluster together"
    elif avg_corr > 0.3:
        summary_text += "→ Moderate correlation\n"
        summary_text += "→ Some clustering by luminosity"
    else:
        summary_text += "→ Weak correlation\n"
        summary_text += "→ Luminosities are independent"
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=13, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'luminosity_correlations.png', dpi=300, bbox_inches='tight')
    print(f"Saved: luminosity_correlations.png")
    plt.close()

def plot_ternary_diagram(metrics, output_dir):
    """
    Ternary-like diagram showing L_max, L_mid, L_min relationships.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize luminosities to sum to 1
    L_total = metrics['L_max'] + metrics['L_mid'] + metrics['L_min']
    L_max_frac = metrics['L_max'] / L_total
    L_mid_frac = metrics['L_mid'] / L_total
    L_min_frac = metrics['L_min'] / L_total
    
    # Plot on 2D: x = L_max fraction, y = L_mid fraction
    # (L_min is implicitly 1 - x - y)
    scatter = ax.scatter(L_max_frac, L_mid_frac, 
                        c=np.log10(metrics['L_max']), 
                        cmap='viridis', s=50, alpha=0.6, 
                        edgecolors='k', linewidth=0.5)
    
    # Add reference lines
    # Equal luminosities: L_max = L_mid = L_min = 1/3
    ax.plot(1/3, 1/3, 'r*', markersize=20, label='Equal luminosities')
    
    # Dominant L_max: high x, low y
    ax.annotate('Dominant\nBrightest', xy=(0.8, 0.1), fontsize=10,
                ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Equal bright pair: moderate x, high y
    ax.annotate('Bright\nPair', xy=(0.4, 0.5), fontsize=10,
                ha='center', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log₁₀(L_max) [erg/s]', fontsize=11)
    
    ax.set_xlabel('L_max / L_total (Brightest Fraction)', fontsize=12)
    ax.set_ylabel('L_mid / L_total (Middle Fraction)', fontsize=12)
    ax.set_title('Luminosity Distribution Within Triple Systems', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(0.33, 1.0)
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'luminosity_ternary.png', dpi=300, bbox_inches='tight')
    print(f"Saved: luminosity_ternary.png")
    plt.close()

def plot_hierarchy_vs_separation(df, metrics, output_dir):
    """
    Does luminosity hierarchy correlate with spatial configuration?
    """
    # Need separation data from original df
    separations = pd.DataFrame({
        'd01': df['Separation_3D_1_kpc_com'].values,
        'd02': df['Separation_3D_2_kpc_com'].values,
        'd12': df['Separation_3D_12_kpc_com'].values
    })
    
    sorted_seps = np.sort(separations.values, axis=1)
    d_min = sorted_seps[:, 0]
    d_max = sorted_seps[:, 2]
    spatial_compactness = d_max / d_min
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Luminosity spread vs spatial compactness
    ax = axes[0]
    scatter = ax.scatter(spatial_compactness, metrics['lum_spread_dex'],
                        c=np.log10(metrics['L_max']), cmap='plasma',
                        s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.set_xlabel('Spatial Compactness (d_max/d_min)', fontsize=12)
    ax.set_ylabel('Luminosity Spread (log₁₀(L_max/L_min)) [dex]', fontsize=12)
    ax.set_title('Spatial vs Luminosity Hierarchy', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log₁₀(L_max) [erg/s]', fontsize=10)
    
    # Compute correlation
    corr_spatial_lum = np.corrcoef(spatial_compactness, metrics['lum_spread_dex'])[0, 1]
    ax.text(0.05, 0.95, f'r = {corr_spatial_lum:.3f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Composition by spatial type
    ax = axes[1]
    
    # Define spatial types
    hierarchical_spatial = spatial_compactness > 2.5
    compact_spatial = spatial_compactness < 2.0
    
    # Count compositions for each spatial type
    comp_hier = metrics[hierarchical_spatial]['composition'].value_counts().head(5)
    comp_compact = metrics[compact_spatial]['composition'].value_counts().head(5)
    
    x = np.arange(max(len(comp_hier), len(comp_compact)))
    width = 0.35
    
    # Get top compositions overall
    all_comps = metrics['composition'].value_counts().head(5).index
    
    hier_counts = [comp_hier.get(c, 0) for c in all_comps]
    compact_counts = [comp_compact.get(c, 0) for c in all_comps]
    
    ax.bar(x - width/2, hier_counts, width, label='Hierarchical spatial', 
           color='red', alpha=0.7)
    ax.bar(x + width/2, compact_counts, width, label='Compact spatial', 
           color='blue', alpha=0.7)
    
    ax.set_xlabel('Luminosity Composition', fontsize=12)
    ax.set_ylabel('Number of Systems', fontsize=12)
    ax.set_title('Lum. Composition by Spatial Configuration', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_comps, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'hierarchy_vs_separation.png', dpi=300, bbox_inches='tight')
    print(f"Saved: hierarchy_vs_separation.png")
    plt.close()

def plot_luminosity_rank_comparison(metrics, output_dir):
    """
    Compare whether brightest, middle, or faintest have different properties.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Box plot of luminosities by rank
    ax = axes[0, 0]
    data_to_plot = [np.log10(metrics['L_max']), 
                    np.log10(metrics['L_mid']), 
                    np.log10(metrics['L_min'])]
    bp = ax.boxplot(data_to_plot, labels=['Brightest', 'Middle', 'Faintest'],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], ['red', 'orange', 'blue']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('log₁₀(Lbol) [erg/s]', fontsize=12)
    ax.set_title('Luminosity by Rank', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ratio of brightest to middle
    ax = axes[0, 1]
    ax.hist(np.log10(metrics['ratio_max_mid']), bins=25, 
            color='green', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Equal (ratio=1)')
    ax.set_xlabel('log₁₀(L_max / L_mid)', fontsize=12)
    ax.set_ylabel('Number of Systems', fontsize=12)
    ax.set_title('Brightest vs Middle AGN', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # What fraction have similar luminosities?
    similar_threshold = 0.3  # within 0.3 dex = factor of 2
    frac_similar_max_mid = (np.abs(np.log10(metrics['ratio_max_mid'])) < similar_threshold).mean()
    frac_similar_mid_min = (np.abs(np.log10(metrics['ratio_mid_min'])) < similar_threshold).mean()
    
    ax.text(0.98, 0.97, f'{frac_similar_max_mid*100:.1f}% within factor 2',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Ratio of middle to faintest
    ax = axes[1, 0]
    ax.hist(np.log10(metrics['ratio_mid_min']), bins=25, 
            color='cyan', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Equal (ratio=1)')
    ax.set_xlabel('log₁₀(L_mid / L_min)', fontsize=12)
    ax.set_ylabel('Number of Systems', fontsize=12)
    ax.set_title('Middle vs Faintest AGN', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.text(0.98, 0.97, f'{frac_similar_mid_min*100:.1f}% within factor 2',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
    
    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = "Luminosity Rank Statistics\n\n"
    summary += f"Brightest (L_max):\n"
    summary += f"  Median: {10**np.log10(metrics['L_max']).median():.2e} erg/s\n"
    summary += f"  Range: {10**np.log10(metrics['L_max']).min():.2e} - {10**np.log10(metrics['L_max']).max():.2e}\n\n"
    
    summary += f"Middle (L_mid):\n"
    summary += f"  Median: {10**np.log10(metrics['L_mid']).median():.2e} erg/s\n"
    summary += f"  Range: {10**np.log10(metrics['L_mid']).min():.2e} - {10**np.log10(metrics['L_mid']).max():.2e}\n\n"
    
    summary += f"Faintest (L_min):\n"
    summary += f"  Median: {10**np.log10(metrics['L_min']).median():.2e} erg/s\n"
    summary += f"  Range: {10**np.log10(metrics['L_min']).min():.2e} - {10**np.log10(metrics['L_min']).max():.2e}\n\n"
    
    summary += f"Typical Ratios:\n"
    summary += f"  L_max/L_mid: {metrics['ratio_max_mid'].median():.2f}\n"
    summary += f"  L_mid/L_min: {metrics['ratio_mid_min'].median():.2f}\n"
    summary += f"  L_max/L_min: {metrics['ratio_max_min'].median():.2f}\n"
    
    ax.text(0.5, 0.5, summary, transform=ax.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
            family='monospace')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'luminosity_rank_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: luminosity_rank_comparison.png")
    plt.close()

def plot_clustering_analysis(metrics, output_dir):
    """
    Determine if high-L AGN cluster together and low-L AGN cluster together.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 2D histogram: L_brightest vs L_faintest
    ax = axes[0, 0]
    h = ax.hist2d(np.log10(metrics['L_max']), np.log10(metrics['L_min']),
                  bins=20, cmap='YlOrRd', cmin=1)
    plt.colorbar(h[3], ax=ax, label='Number of Systems')
    ax.plot([41, 46], [41, 46], 'b--', linewidth=2, alpha=0.7, label='Equal luminosities')
    ax.set_xlabel('log₁₀(L_max) [erg/s]', fontsize=12)
    ax.set_ylabel('log₁₀(L_min) [erg/s]', fontsize=12)
    ax.set_title('Brightest vs Faintest Clustering', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Check if diagonal or off-diagonal
    corr_max_min = np.corrcoef(np.log10(metrics['L_max']), 
                                np.log10(metrics['L_min']))[0, 1]
    
    if corr_max_min > 0.3:
        interp = "Positive correlation:\nHigh-L systems have\nhigh-L companions"
    else:
        interp = "Weak correlation:\nLuminosities are\nindependent"
    
    ax.text(0.05, 0.95, f'r = {corr_max_min:.3f}\n\n{interp}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Compare class compositions
    ax = axes[0, 1]
    
    # How many systems have all high-L vs all low-L?
    all_high = (metrics['n_high'] == 3).sum()
    all_medium = (metrics['n_medium'] == 3).sum()
    all_low = (metrics['n_low'] == 3).sum()
    mixed = len(metrics) - all_high - all_medium - all_low
    
    categories = ['All High\n(3H)', 'All Medium\n(3M)', 'All Low\n(3L)', 'Mixed']
    counts = [all_high, all_medium, all_low, mixed]
    colors_bar = ['red', 'orange', 'blue', 'green']
    
    bars = ax.bar(categories, counts, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Systems', fontsize=12)
    ax.set_title('Luminosity Homogeneity', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({100*count/len(metrics):.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    # Distribution of number of high-L AGN per system
    ax = axes[1, 0]
    n_high_counts = metrics['n_high'].value_counts().sort_index()
    ax.bar(n_high_counts.index, n_high_counts.values, 
           color='red', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of High-L AGN per System', fontsize=12)
    ax.set_ylabel('Number of Systems', fontsize=12)
    ax.set_title('Distribution of High Luminosity AGN', fontsize=13, fontweight='bold')
    ax.set_xticks([0, 1, 2, 3])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentages
    for x, y in zip(n_high_counts.index, n_high_counts.values):
        ax.text(x, y, f'{100*y/len(metrics):.1f}%', 
                ha='center', va='bottom', fontsize=11)
    
    # Stacked bar showing composition patterns
    ax = axes[1, 1]
    
    # Top 8 compositions
    top_comps = metrics['composition'].value_counts().head(8)
    
    ax.barh(range(len(top_comps)), top_comps.values, color='steelblue', 
            alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(top_comps)))
    ax.set_yticklabels(top_comps.index, fontsize=10)
    ax.set_xlabel('Number of Systems', fontsize=12)
    ax.set_title('Top Luminosity Compositions', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add counts and percentages
    for i, (comp, count) in enumerate(top_comps.items()):
        ax.text(count, i, f'  {count} ({100*count/len(metrics):.1f}%)', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'clustering_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: clustering_analysis.png")
    plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
def print_summary_statistics(metrics):
    """Print comprehensive summary of luminosity patterns."""
    print("\n" + "="*70)
    print("TRIPLE AGN LUMINOSITY ANALYSIS")
    print("="*70)
    
    print(f"\nTotal triple systems analyzed: {len(metrics)}")
    
    print("\n--- Luminosity Ranges (all AGN) ---")
    all_lums = np.concatenate([metrics['L0'], metrics['L1'], metrics['L2']])
    print(f"Minimum:   {all_lums.min():.2e} erg/s (log = {np.log10(all_lums.min()):.2f})")
    print(f"Median:    {np.median(all_lums):.2e} erg/s (log = {np.log10(np.median(all_lums)):.2f})")
    print(f"Maximum:   {all_lums.max():.2e} erg/s (log = {np.log10(all_lums.max()):.2f})")
    print(f"Mean:      {all_lums.mean():.2e} erg/s (log = {np.log10(all_lums.mean()):.2f})")
    
    print("\n--- Luminosity by Rank ---")
    print(f"Brightest (L_max):")
    print(f"  Median: {metrics['L_max'].median():.2e} erg/s")
    print(f"  Mean:   {metrics['L_max'].mean():.2e} erg/s")
    
    print(f"\nMiddle (L_mid):")
    print(f"  Median: {metrics['L_mid'].median():.2e} erg/s")
    print(f"  Mean:   {metrics['L_mid'].mean():.2e} erg/s")
    
    print(f"\nFaintest (L_min):")
    print(f"  Median: {metrics['L_min'].median():.2e} erg/s")
    print(f"  Mean:   {metrics['L_min'].mean():.2e} erg/s")
    
    print("\n--- Luminosity Ratios ---")
    print(f"L_max / L_mid:")
    print(f"  Median: {metrics['ratio_max_mid'].median():.2f}")
    print(f"  Mean:   {metrics['ratio_max_mid'].mean():.2f}")
    
    print(f"\nL_mid / L_min:")
    print(f"  Median: {metrics['ratio_mid_min'].median():.2f}")
    print(f"  Mean:   {metrics['ratio_mid_min'].mean():.2f}")
    
    print(f"\nL_max / L_min:")
    print(f"  Median: {metrics['ratio_max_min'].median():.2f}")
    print(f"  Mean:   {metrics['ratio_max_min'].mean():.2f}")
    
    print("\n--- Luminosity Spread ---")
    print(f"Log₁₀(L_max/L_min) in dex:")
    print(f"  Median: {metrics['lum_spread_dex'].median():.2f} dex")
    print(f"  Mean:   {metrics['lum_spread_dex'].mean():.2f} dex")
    print(f"  Range:  {metrics['lum_spread_dex'].min():.2f} - {metrics['lum_spread_dex'].max():.2f} dex")
    
    print("\n--- Luminosity Class Composition ---")
    n_high_counts = metrics['n_high'].value_counts().sort_index()
    for n_high, count in n_high_counts.items():
        print(f"{n_high} High-L AGN: {count} systems ({100*count/len(metrics):.1f}%)")
    
    print("\n--- Top Luminosity Patterns ---")
    top_comps = metrics['composition'].value_counts().head(5)
    for comp, count in top_comps.items():
        print(f"{comp}: {count} systems ({100*count/len(metrics):.1f}%)")
    
    print("\n--- Clustering Analysis ---")
    # Correlation between BH luminosities
    corr01 = np.corrcoef(np.log10(metrics['L0']), np.log10(metrics['L1']))[0, 1]
    corr02 = np.corrcoef(np.log10(metrics['L0']), np.log10(metrics['L2']))[0, 1]
    corr12 = np.corrcoef(np.log10(metrics['L1']), np.log10(metrics['L2']))[0, 1]
    avg_corr = np.mean([corr01, corr02, corr12])
    
    print(f"Average pairwise correlation: r = {avg_corr:.3f}")
    
    if avg_corr > 0.5:
        print("→ STRONG clustering: High-L AGN tend to group with high-L AGN")
        print("→ Low-L AGN tend to group with low-L AGN")
    elif avg_corr > 0.3:
        print("→ MODERATE clustering: Some tendency for similar luminosities")
    else:
        print("→ WEAK clustering: Luminosities appear independent")
    
    print("\n--- Homogeneity ---")
    all_high = (metrics['n_high'] == 3).sum()
    all_similar = ((metrics['n_high'] == 3) | (metrics['n_medium'] == 3) | 
                   (metrics['n_low'] == 3)).sum()
    
    print(f"All three AGN in same class: {all_similar} ({100*all_similar/len(metrics):.1f}%)")
    print(f"All three high-L: {all_high} ({100*all_high/len(metrics):.1f}%)")
    
    print("\n" + "="*70)
    
    # Answer the key questions
    print("\nKEY FINDINGS:")
    print("-" * 70)
    
    # Question 1: Do high-L cluster together?
    if avg_corr > 0.3:
        print("✓ HIGH-L AGN CLUSTER TOGETHER (positive correlation)")
        print("✓ LOW-L AGN CLUSTER TOGETHER")
    else:
        print("✓ Luminosities appear INDEPENDENT across triple members")
    
    # Question 2: Most common pattern
    one_bright_two_dim = metrics[(metrics['n_high'] == 1) & 
                                 (metrics['n_medium'] + metrics['n_low'] == 2)].shape[0]
    two_bright_one_dim = metrics[(metrics['n_high'] == 2) & 
                                 (metrics['n_medium'] + metrics['n_low'] == 1)].shape[0]
    
    print(f"\n✓ 1 bright + 2 dim: {one_bright_two_dim} systems ({100*one_bright_two_dim/len(metrics):.1f}%)")
    print(f"✓ 2 bright + 1 dim: {two_bright_one_dim} systems ({100*two_bright_one_dim/len(metrics):.1f}%)")
    
    if one_bright_two_dim > two_bright_one_dim:
        print("→ MOST COMMON: One high-L AGN with two lower-L companions")
    elif two_bright_one_dim > one_bright_two_dim:
        print("→ MOST COMMON: Two high-L AGN with one lower-L companion")
    else:
        print("→ NO CLEAR PREFERENCE between these patterns")
    
    # Question 3: Luminosity hierarchy strength
    if metrics['lum_spread_dex'].median() > 1.0:
        print(f"\n✓ STRONG luminosity hierarchies (median spread: {metrics['lum_spread_dex'].median():.2f} dex)")
        print("→ Brightest often >10x brighter than faintest")
    else:
        print(f"\n✓ MODERATE luminosity hierarchies (median spread: {metrics['lum_spread_dex'].median():.2f} dex)")
    
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
    
    print("\nComputing luminosity metrics...")
    metrics = compute_luminosity_metrics(df)
    
    print("\nGenerating plots...")
    plot_luminosity_distributions(metrics, OUTPUT_DIR)
    plot_composition_pie_chart(metrics, OUTPUT_DIR)
    plot_luminosity_correlation_matrix(metrics, OUTPUT_DIR)
    plot_ternary_diagram(metrics, OUTPUT_DIR)
    plot_hierarchy_vs_separation(df, metrics, OUTPUT_DIR)
    plot_luminosity_rank_comparison(metrics, OUTPUT_DIR)
    plot_clustering_analysis(metrics, OUTPUT_DIR)
    
    print_summary_statistics(metrics)
    
    # Save metrics to file
    metrics_path = Path(OUTPUT_DIR) / 'triple_agn_luminosity_metrics.csv'
    metrics.to_csv(metrics_path, index=False)
    print(f"\nSaved luminosity metrics to: {metrics_path}")
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()