#!/usr/bin/env python3
"""
create_gaia_summary.py

Create a comprehensive summary of Gaia data broken down by distances from the galaxy center.
This script processes raw Gaia data and generates statistics for different regions of the Milky Way.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import Galactocentric
import warnings
warnings.filterwarnings('ignore')

class GaiaDataProcessor:
    """Process raw Gaia data and calculate galactocentric properties."""
    
    def __init__(self):
        # Solar position and motion parameters (from recent measurements)
        self.R_sun = 8.122  # kpc (GRAVITY Collaboration 2019)
        self.z_sun = 0.0208  # kpc
        self.v_sun = 232.8  # km/s (Reid & Brunthaler 2004)
        
        # Galactocentric frame parameters
        self.galactocentric_frame = Galactocentric(
            galcen_distance=self.R_sun * u.kpc,
            z_sun=self.z_sun * u.kpc
        )
    
    def process_gaia_data(self, data_path):
        """Process raw Gaia data to calculate distances and velocities."""
        print(f"Loading Gaia data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} stars")
        
        # Filter for good quality data
        print("Filtering for good quality data...")
        mask = (
            (df['parallax'] > 0) &  # Positive parallax
            (df['parallax_error'] / df['parallax'] < 0.2) &  # Good parallax precision
            (df['ruwe'] < 1.4) &  # Good astrometric fit
            (df['radial_velocity'].notna()) &  # Has radial velocity
            (df['pmra'].notna()) & (df['pmdec'].notna())  # Has proper motions
        )
        df_filtered = df[mask].copy()
        print(f"After filtering: {len(df_filtered)} stars")
        
        # Calculate distances
        print("Calculating distances...")
        df_filtered['distance'] = 1000.0 / df_filtered['parallax']  # pc
        df_filtered['distance_kpc'] = df_filtered['distance'] / 1000.0
        
        # Convert coordinates to radians
        ra_rad = np.radians(df_filtered['ra'])
        dec_rad = np.radians(df_filtered['dec'])
        l_rad = np.radians(df_filtered['l'])
        b_rad = np.radians(df_filtered['b'])
        
        # Calculate galactocentric distances (simplified)
        # Using the fact that we're near the Galactic plane
        print("Calculating galactocentric coordinates...")
        
        # For stars near the Sun, approximate R using distance and Galactic longitude
        # R² = R_sun² + d² - 2*R_sun*d*cos(l)
        cos_l = np.cos(l_rad)
        d = df_filtered['distance_kpc']
        
        df_filtered['R_kpc'] = np.sqrt(self.R_sun**2 + d**2 - 2*self.R_sun*d*cos_l)
        
        # Calculate z coordinate (height above plane)
        df_filtered['z_kpc'] = d * np.sin(b_rad)
        
        # Calculate velocities (simplified)
        # Convert proper motions to tangential velocities
        df_filtered['v_tangential'] = np.sqrt(
            (df_filtered['pmra'] * df_filtered['distance_kpc'] * 4.74)**2 +
            (df_filtered['pmdec'] * df_filtered['distance_kpc'] * 4.74)**2
        )
        
        # Approximate circular velocity from tangential velocity
        df_filtered['v_circ'] = df_filtered['v_tangential']
        
        # Use radial velocity directly
        df_filtered['v_R'] = df_filtered['radial_velocity']
        
        # Calculate total velocity
        df_filtered['v_total'] = np.sqrt(
            df_filtered['v_R']**2 + df_filtered['v_tangential']**2
        )
        
        # Calculate velocity dispersions for error estimation
        df_filtered['sigma_v'] = np.sqrt(
            (df_filtered['radial_velocity_error'])**2 +
            (df_filtered['pmra_error'] * df_filtered['distance_kpc'] * 4.74)**2 +
            (df_filtered['pmdec_error'] * df_filtered['distance_kpc'] * 4.74)**2
        )
        
        return df_filtered
    
    def create_distance_bins(self, df, bin_edges=None):
        """Create distance bins and calculate statistics for each bin."""
        if bin_edges is None:
            # Create bins from 0 to 30 kpc with varying bin sizes
            bin_edges = np.array([
                0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30
            ])
        
        print(f"Creating distance bins: {bin_edges}")
        
        # Create bins
        df['distance_bin'] = pd.cut(df['R_kpc'], bins=bin_edges, labels=False)
        
        # Calculate statistics for each bin
        bin_stats = []
        
        for i in range(len(bin_edges) - 1):
            mask = df['distance_bin'] == i
            bin_data = df[mask]
            
            if len(bin_data) == 0:
                continue
            
            # Basic statistics
            stats = {
                'bin_index': i,
                'R_min': bin_edges[i],
                'R_max': bin_edges[i + 1],
                'R_center': (bin_edges[i] + bin_edges[i + 1]) / 2,
                'n_stars': len(bin_data),
                'volume_kpc3': self.calculate_volume(bin_edges[i], bin_edges[i + 1]),
                'density_stars_kpc3': len(bin_data) / self.calculate_volume(bin_edges[i], bin_edges[i + 1])
            }
            
            # Velocity statistics
            if len(bin_data) > 10:  # Only calculate if enough stars
                stats.update({
                    'v_circ_mean': bin_data['v_circ'].mean(),
                    'v_circ_std': bin_data['v_circ'].std(),
                    'v_circ_median': bin_data['v_circ'].median(),
                    'v_R_mean': bin_data['v_R'].mean(),
                    'v_R_std': bin_data['v_R'].std(),
                    'v_tangential_mean': bin_data['v_tangential'].mean(),
                    'v_tangential_std': bin_data['v_tangential'].std(),
                    'v_total_mean': bin_data['v_total'].mean(),
                    'v_total_std': bin_data['v_total'].std(),
                    'sigma_v_mean': bin_data['sigma_v'].mean(),
                    'sigma_v_median': bin_data['sigma_v'].median()
                })
                
                # Height statistics
                stats.update({
                    'z_mean': bin_data['z_kpc'].mean(),
                    'z_std': bin_data['z_kpc'].std(),
                    'z_median': bin_data['z_kpc'].median(),
                    'z_abs_mean': np.abs(bin_data['z_kpc']).mean()
                })
                
                # Magnitude statistics (brightness)
                stats.update({
                    'mag_mean': bin_data['phot_g_mean_mag'].mean(),
                    'mag_std': bin_data['phot_g_mean_mag'].std(),
                    'mag_median': bin_data['phot_g_mean_mag'].median()
                })
            
            bin_stats.append(stats)
        
        return pd.DataFrame(bin_stats)
    
    def calculate_volume(self, R_min, R_max):
        """Calculate volume of a cylindrical shell."""
        # Assume disk height of 0.5 kpc for volume calculation
        h_disk = 0.5  # kpc
        volume = np.pi * (R_max**2 - R_min**2) * h_disk
        return volume
    
    def create_detailed_summary(self, df, output_path):
        """Create a detailed summary file with all statistics."""
        print("Creating detailed summary...")
        
        # Create distance bins
        bin_stats = self.create_distance_bins(df)
        
        # Add additional calculations
        bin_stats['v_circ_km_s'] = bin_stats['v_circ_mean']  # For clarity
        bin_stats['density_relative'] = bin_stats['density_stars_kpc3'] / bin_stats['density_stars_kpc3'].max()
        
        # Calculate rotation curve slope
        bin_stats['dv_dr'] = np.gradient(bin_stats['v_circ_mean'], bin_stats['R_center'])
        
        # Save detailed summary
        bin_stats.to_csv(output_path, index=False)
        print(f"Saved detailed summary to {output_path}")
        
        return bin_stats
    
    def create_visualizations(self, df, bin_stats, output_prefix):
        """Create visualizations of the data distribution."""
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Star count vs distance
        ax = axes[0, 0]
        ax.bar(bin_stats['R_center'], bin_stats['n_stars'], 
               width=bin_stats['R_max'] - bin_stats['R_min'], alpha=0.7)
        ax.set_xlabel('R (kpc)')
        ax.set_ylabel('Number of Stars')
        ax.set_title('Star Count vs Distance from Galactic Center')
        ax.grid(True, alpha=0.3)
        
        # 2. Rotation curve
        ax = axes[0, 1]
        ax.errorbar(bin_stats['R_center'], bin_stats['v_circ_mean'], 
                   yerr=bin_stats['v_circ_std'], fmt='o-', capsize=3)
        ax.set_xlabel('R (kpc)')
        ax.set_ylabel('V_circ (km/s)')
        ax.set_title('Rotation Curve')
        ax.grid(True, alpha=0.3)
        
        # 3. Star density vs distance
        ax = axes[0, 2]
        ax.semilogy(bin_stats['R_center'], bin_stats['density_stars_kpc3'], 'o-')
        ax.set_xlabel('R (kpc)')
        ax.set_ylabel('Star Density (stars/kpc³)')
        ax.set_title('Star Density vs Distance')
        ax.grid(True, alpha=0.3)
        
        # 4. Velocity dispersions
        ax = axes[1, 0]
        ax.plot(bin_stats['R_center'], bin_stats['v_R_std'], 'o-', label='σ_vR')
        ax.plot(bin_stats['R_center'], bin_stats['v_tangential_std'], 's-', label='σ_v_tangential')
        ax.set_xlabel('R (kpc)')
        ax.set_ylabel('Velocity Dispersion (km/s)')
        ax.set_title('Velocity Dispersions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Height distribution
        ax = axes[1, 1]
        ax.plot(bin_stats['R_center'], bin_stats['z_abs_mean'], 'o-')
        ax.set_xlabel('R (kpc)')
        ax.set_ylabel('Mean |z| (kpc)')
        ax.set_title('Height Distribution')
        ax.grid(True, alpha=0.3)
        
        # 6. Magnitude distribution
        ax = axes[1, 2]
        ax.plot(bin_stats['R_center'], bin_stats['mag_mean'], 'o-')
        ax.set_xlabel('R (kpc)')
        ax.set_ylabel('Mean G Magnitude')
        ax.set_title('Brightness Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'../plots/{output_prefix}_summary_plots.png', dpi=150, bbox_inches='tight')
        print(f"Saved visualizations to ../plots/{output_prefix}_summary_plots.png")
        
        return fig

def main():
    """Main execution function."""
    print("="*60)
    print("GAIA DATA SUMMARY GENERATOR")
    print("="*60)
    
    # Initialize processor
    processor = GaiaDataProcessor()
    
    # Process Gaia data
    data_path = '../gaia_sky_slices/all_sky_gaia.csv'
    df_processed = processor.process_gaia_data(data_path)
    
    # Create summary
    summary_path = '../data/gaia_summary/gaia_distance_summary.csv'
    bin_stats = processor.create_detailed_summary(df_processed, summary_path)
    
    # Create visualizations
    processor.create_visualizations(df_processed, bin_stats, 'gaia')
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total stars processed: {len(df_processed)}")
    print(f"Distance range: {df_processed['R_kpc'].min():.1f} - {df_processed['R_kpc'].max():.1f} kpc")
    print(f"Velocity range: {df_processed['v_circ'].min():.1f} - {df_processed['v_circ'].max():.1f} km/s")
    
    print("\nDistance bin statistics:")
    print(bin_stats[['R_center', 'n_stars', 'v_circ_mean', 'density_stars_kpc3']].to_string(index=False))
    
    # Save processed data for future use
    df_processed.to_csv('../data/gaia_processed/gaia_processed_data.csv', index=False)
    print(f"\nSaved processed data to ../data/gaia_processed/gaia_processed_data.csv")
    
    print("\n" + "="*60)
    print("SUMMARY GENERATION COMPLETE!")
    print("="*60)
    print(f"Files created:")
    print(f"  - {summary_path}: Distance bin statistics")
    print(f"  - ../data/gaia_processed/gaia_processed_data.csv: Full processed dataset")
    print(f"  - ../plots/gaia_summary_plots.png: Visualizations")

if __name__ == '__main__':
    main() 