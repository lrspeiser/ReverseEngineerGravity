#!/usr/bin/env python3
"""
analyze_gaia_summary.py

Analyze the Gaia distance summary data and provide insights about the distribution
of stars across different regions of the Milky Way.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_gaia_summary(summary_file='../data/gaia_summary/gaia_distance_summary.csv'):
    """Analyze the Gaia summary data and print insights."""
    
    print("="*80)
    print("GAIA DATA DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Load the summary data
    df = pd.read_csv(summary_file)
    
    # Filter out bins with too few stars for meaningful statistics
    df_clean = df[df['n_stars'] > 10].copy()
    
    print(f"Total distance bins analyzed: {len(df_clean)}")
    print(f"Total stars in analysis: {df_clean['n_stars'].sum():,}")
    print(f"Distance range: {df_clean['R_center'].min():.1f} - {df_clean['R_center'].max():.1f} kpc")
    
    print("\n" + "="*80)
    print("KEY FINDINGS BY GALACTIC REGION")
    print("="*80)
    
    # Analyze each region
    for _, row in df_clean.iterrows():
        R_center = row['R_center']
        n_stars = row['n_stars']
        v_circ = row['v_circ_mean']
        density = row['density_stars_kpc3']
        z_height = row['z_abs_mean']
        mag = row['mag_mean']
        
        print(f"\n📍 GALACTIC RADIUS {R_center:.1f} kpc:")
        print(f"   • {n_stars:,} stars in this region")
        print(f"   • Star density: {density:.1f} stars/kpc³")
        print(f"   • Mean circular velocity: {v_circ:.1f} km/s")
        print(f"   • Mean height above plane: {z_height:.3f} kpc")
        print(f"   • Mean brightness (G magnitude): {mag:.2f}")
        
        # Add insights based on location
        if R_center < 6:
            print(f"   💡 INNER GALAXY: High velocity, low star count (selection effects)")
        elif R_center < 10:
            print(f"   💡 SOLAR NEIGHBORHOOD: Peak star density, well-sampled region")
        elif R_center < 15:
            print(f"   💡 OUTER DISK: Declining density, still significant population")
        else:
            print(f"   💡 FAR OUTER REGION: Very sparse, likely halo stars")
    
    print("\n" + "="*80)
    print("OVERALL PATTERNS")
    print("="*80)
    
    # Star density analysis
    max_density_bin = df_clean.loc[df_clean['density_stars_kpc3'].idxmax()]
    print(f"📊 PEAK STAR DENSITY:")
    print(f"   • Location: {max_density_bin['R_center']:.1f} kpc from center")
    print(f"   • Density: {max_density_bin['density_stars_kpc3']:.0f} stars/kpc³")
    print(f"   • {max_density_bin['n_stars']:,} stars in this bin")
    
    # Velocity analysis
    print(f"\n📊 ROTATION CURVE CHARACTERISTICS:")
    print(f"   • Velocity range: {df_clean['v_circ_mean'].min():.1f} - {df_clean['v_circ_mean'].max():.1f} km/s")
    print(f"   • Mean velocity: {df_clean['v_circ_mean'].mean():.1f} km/s")
    
    # Check for flat rotation curve
    velocity_gradient = np.gradient(df_clean['v_circ_mean'], df_clean['R_center'])
    flat_rotation = np.all(np.abs(velocity_gradient) < 5)  # km/s/kpc
    print(f"   • Rotation curve appears {'FLAT' if flat_rotation else 'VARIABLE'}")
    
    # Height analysis
    print(f"\n📊 VERTICAL STRUCTURE:")
    print(f"   • Mean height range: {df_clean['z_abs_mean'].min():.3f} - {df_clean['z_abs_mean'].max():.3f} kpc")
    print(f"   • Disk thickness appears {'THIN' if df_clean['z_abs_mean'].mean() < 0.2 else 'THICK'}")
    
    # Brightness analysis
    print(f"\n📊 BRIGHTNESS DISTRIBUTION:")
    print(f"   • Magnitude range: {df_clean['mag_mean'].min():.2f} - {df_clean['mag_mean'].max():.2f}")
    print(f"   • Mean brightness: {df_clean['mag_mean'].mean():.2f} (G magnitude)")
    
    # Data quality assessment
    print(f"\n📊 DATA QUALITY:")
    total_stars = df_clean['n_stars'].sum()
    inner_stars = df_clean[df_clean['R_center'] < 8]['n_stars'].sum()
    outer_stars = df_clean[df_clean['R_center'] >= 8]['n_stars'].sum()
    
    print(f"   • Inner galaxy (<8 kpc): {inner_stars:,} stars ({inner_stars/total_stars*100:.1f}%)")
    print(f"   • Outer galaxy (≥8 kpc): {outer_stars:,} stars ({outer_stars/total_stars*100:.1f}%)")
    print(f"   • Sampling bias: {'STRONG' if inner_stars/outer_stars > 10 else 'MODERATE' if inner_stars/outer_stars > 3 else 'MINIMAL'}")
    
    return df_clean

def create_detailed_plots(df, output_prefix='gaia_analysis'):
    """Create detailed analysis plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Star count distribution
    ax = axes[0, 0]
    ax.bar(df['R_center'], df['n_stars'], alpha=0.7, color='skyblue')
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('Number of Stars')
    ax.set_title('Star Count Distribution')
    ax.grid(True, alpha=0.3)
    
    # 2. Star density profile
    ax = axes[0, 1]
    ax.semilogy(df['R_center'], df['density_stars_kpc3'], 'o-', color='red', linewidth=2)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('Star Density (stars/kpc³)')
    ax.set_title('Star Density Profile')
    ax.grid(True, alpha=0.3)
    
    # 3. Rotation curve
    ax = axes[0, 2]
    ax.errorbar(df['R_center'], df['v_circ_mean'], yerr=df['v_circ_std'], 
                fmt='o-', capsize=3, color='green', linewidth=2)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('Circular Velocity (km/s)')
    ax.set_title('Rotation Curve')
    ax.grid(True, alpha=0.3)
    
    # 4. Height distribution
    ax = axes[1, 0]
    ax.plot(df['R_center'], df['z_abs_mean'], 'o-', color='purple', linewidth=2)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('Mean |z| (kpc)')
    ax.set_title('Vertical Structure')
    ax.grid(True, alpha=0.3)
    
    # 5. Velocity dispersions
    ax = axes[1, 1]
    ax.plot(df['R_center'], df['v_R_std'], 'o-', label='Radial (σ_vR)', color='orange')
    ax.plot(df['R_center'], df['v_tangential_std'], 's-', label='Tangential (σ_vφ)', color='brown')
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('Velocity Dispersion (km/s)')
    ax.set_title('Velocity Dispersions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Brightness distribution
    ax = axes[1, 2]
    ax.plot(df['R_center'], df['mag_mean'], 'o-', color='darkblue', linewidth=2)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('Mean G Magnitude')
    ax.set_title('Brightness Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../plots/{output_prefix}_detailed_plots.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 Saved detailed analysis plots to ../plots/{output_prefix}_detailed_plots.png")
    
    return fig

def generate_insights_report(df):
    """Generate a comprehensive insights report."""
    
    report = []
    report.append("="*80)
    report.append("GAIA DATA INSIGHTS REPORT")
    report.append("="*80)
    report.append("")
    
    # Overall statistics
    report.append("📊 OVERALL STATISTICS:")
    report.append(f"   • Total stars analyzed: {df['n_stars'].sum():,}")
    report.append(f"   • Distance range: {df['R_center'].min():.1f} - {df['R_center'].max():.1f} kpc")
    report.append(f"   • Number of distance bins: {len(df)}")
    report.append("")
    
    # Galactic structure insights
    report.append("🌌 GALACTIC STRUCTURE INSIGHTS:")
    
    # Star density peak
    peak_bin = df.loc[df['density_stars_kpc3'].idxmax()]
    report.append(f"   • Peak star density occurs at {peak_bin['R_center']:.1f} kpc")
    report.append(f"   • This suggests the solar neighborhood is well-sampled")
    
    # Rotation curve analysis
    v_gradient = np.gradient(df['v_circ_mean'], df['R_center'])
    if np.all(np.abs(v_gradient) < 5):
        report.append("   • Rotation curve appears relatively flat (consistent with dark matter)")
    else:
        report.append("   • Rotation curve shows significant variation with radius")
    
    # Vertical structure
    mean_height = df['z_abs_mean'].mean()
    if mean_height < 0.2:
        report.append("   • Stars are concentrated in a thin disk")
    else:
        report.append("   • Significant vertical structure detected")
    
    report.append("")
    
    # Data quality assessment
    report.append("🔍 DATA QUALITY ASSESSMENT:")
    
    # Sampling bias
    inner_frac = df[df['R_center'] < 8]['n_stars'].sum() / df['n_stars'].sum()
    if inner_frac > 0.8:
        report.append("   • Strong sampling bias toward inner galaxy")
    elif inner_frac > 0.6:
        report.append("   • Moderate sampling bias toward inner galaxy")
    else:
        report.append("   • Relatively uniform sampling across distances")
    
    # Velocity precision
    mean_sigma = df['sigma_v_mean'].mean()
    if mean_sigma < 10:
        report.append("   • Excellent velocity precision")
    elif mean_sigma < 20:
        report.append("   • Good velocity precision")
    else:
        report.append("   • Moderate velocity precision")
    
    report.append("")
    
    # Scientific implications
    report.append("🔬 SCIENTIFIC IMPLICATIONS:")
    report.append("   • Data suitable for rotation curve analysis")
    report.append("   • Can constrain dark matter distribution")
    report.append("   • Provides insights into Galactic structure")
    report.append("   • Useful for testing modified gravity theories")
    
    report.append("")
    report.append("="*80)
    
    return "\n".join(report)

def main():
    """Main execution function."""
    
    # Analyze the summary data
    df_clean = analyze_gaia_summary()
    
    # Create detailed plots
    create_detailed_plots(df_clean)
    
    # Generate insights report
    report = generate_insights_report(df_clean)
    print(report)
    
    # Save report to file
    with open('../reports/gaia_insights_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n📄 Saved insights report to ../reports/gaia_insights_report.txt")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main() 