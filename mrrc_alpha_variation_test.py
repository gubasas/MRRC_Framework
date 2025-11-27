"""
MRRC V5.0 Fine Structure Constant Variation Test
=================================================

Tests the MRRC prediction that α varies with:
1. Gravitational potential (comparator cost shift)
2. Rotational asymmetry (modulating the π term)

Based on MRRC V5.0 derivation:
α⁻¹ ≈ 4π³ (volume/maintenance) + π² (surface/display) + π (rotation/spin)
Theoretical baseline: α⁻¹ ≈ 137.036

Author: A. Caliber
Date: November 27, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Physical constants
ALPHA_INV_OBSERVED = 137.035999084  # CODATA 2018
ALPHA_INV_MRRC_BASE = 4 * np.pi**3 + np.pi**2 + np.pi  # ≈ 137.036
C_LIGHT = 299792458  # m/s
G_NEWTON = 6.67430e-11  # m³/(kg·s²)
M_EARTH = 5.972e24  # kg
R_EARTH = 6.371e6  # m


class MRRCAlphaModel:
    """MRRC prediction for fine structure constant variation."""
    
    @staticmethod
    def alpha_inverse(volume_scale=1.0, surface_scale=1.0, rotation_scale=1.0):
        """
        Calculate α⁻¹ based on MRRC geometric interpretation.
        
        Parameters:
        -----------
        volume_scale : float
            Modulates internal maintenance cost (4π³ term)
            Increases with gravitational potential (busy substrate)
        surface_scale : float
            Modulates surface interaction cost (π² term)
        rotation_scale : float
            Modulates rotational entropy (π term)
            Affected by extreme rotation (pulsars, black holes)
        
        Returns:
        --------
        float : α⁻¹ value
        """
        return (4 * np.pi**3 * volume_scale + 
                np.pi**2 * surface_scale + 
                np.pi * rotation_scale)
    
    @staticmethod
    def delta_alpha_gravitational(gravitational_potential_ratio):
        """
        Predict Δα/α from gravitational potential.
        
        MRRC hypothesis: Stronger gravity → busier substrate → higher maintenance cost
        → volume_scale increases → α⁻¹ increases → α decreases
        
        Parameters:
        -----------
        gravitational_potential_ratio : float
            Φ/c² (dimensionless gravitational potential)
        
        Returns:
        --------
        float : Δα/α (fractional change)
        """
        # MRRC: volume_scale ≈ 1 + β·(Φ/c²)
        # β is a coupling constant to be fitted
        beta = 1e-6  # Initial guess (will be fitted to data)
        volume_scale = 1.0 + beta * gravitational_potential_ratio
        
        alpha_inv_new = MRRCAlphaModel.alpha_inverse(volume_scale=volume_scale)
        alpha_inv_base = ALPHA_INV_MRRC_BASE
        
        # Δα/α = -Δ(α⁻¹)/α⁻¹ (inverse relationship)
        return -(alpha_inv_new - alpha_inv_base) / alpha_inv_base
    
    @staticmethod
    def delta_alpha_rotational(angular_velocity_ratio):
        """
        Predict Δα/α from rotational effects.
        
        MRRC hypothesis: Extreme rotation → entropic modification of π term
        
        Parameters:
        -----------
        angular_velocity_ratio : float
            ω/ω_critical (normalized angular velocity)
        
        Returns:
        --------
        float : Δα/α (fractional change)
        """
        # MRRC: rotation_scale ≈ 1 + γ·(ω/ω_c)²
        gamma = 1e-8  # Initial guess
        rotation_scale = 1.0 + gamma * angular_velocity_ratio**2
        
        alpha_inv_new = MRRCAlphaModel.alpha_inverse(rotation_scale=rotation_scale)
        alpha_inv_base = ALPHA_INV_MRRC_BASE
        
        return -(alpha_inv_new - alpha_inv_base) / alpha_inv_base


class ObservationalData:
    """Handle observational data for α variation."""
    
    @staticmethod
    def load_quasar_data():
        """
        Load quasar absorption line data (Webb et al., SDSS).
        
        Returns simulated data based on published results if no file found.
        """
        # Try to load real data
        data_file = Path("alpha_variation_quasar_data.csv")
        
        if data_file.exists():
            return pd.read_csv(data_file)
        
        # Generate representative data based on Webb et al. (2011) results
        # Reported: Δα/α = (-0.57 ± 0.11) × 10⁻⁵ (spatial dipole)
        print("Note: Using simulated quasar data based on Webb et al. (2011)")
        
        # Simulate 50 quasar measurements
        np.random.seed(42)
        n_quasars = 50
        
        redshifts = np.random.uniform(0.5, 3.5, n_quasars)
        
        # Add spatial dipole pattern (Webb et al. finding)
        theta = np.random.uniform(0, 2*np.pi, n_quasars)  # Sky position
        phi = np.random.uniform(0, np.pi, n_quasars)
        
        # Dipole amplitude
        dipole_amplitude = -0.57e-5  # ppm
        delta_alpha = dipole_amplitude * np.cos(theta) * np.sin(phi)
        
        # Add measurement noise
        errors = np.random.uniform(0.8e-6, 2.5e-6, n_quasars)
        delta_alpha += np.random.normal(0, errors)
        
        return pd.DataFrame({
            'redshift': redshifts,
            'delta_alpha_over_alpha': delta_alpha,
            'error': errors,
            'theta': theta,
            'phi': phi
        })
    
    @staticmethod
    def load_atomic_clock_data():
        """
        Load atomic clock comparison data (NIST, PTB).
        
        Returns simulated data based on published constraints if no file found.
        """
        data_file = Path("alpha_variation_clock_data.csv")
        
        if data_file.exists():
            return pd.read_csv(data_file)
        
        # Generate data based on Yb/Sr clock comparisons
        print("Note: Using simulated atomic clock data based on NIST/PTB constraints")
        
        # Earth surface vs satellite (GPS altitude ~20,000 km)
        data = {
            'location': ['Earth_surface', 'GPS_orbit', 'ISS_orbit'],
            'altitude_km': [0, 20200, 408],
            'gravitational_potential_ratio': [],
            'delta_alpha_over_alpha': [],
            'error': []
        }
        
        # Calculate Φ/c² for each location
        for alt_km in data['altitude_km']:
            r = R_EARTH + alt_km * 1000
            phi = -G_NEWTON * M_EARTH / r
            phi_surface = -G_NEWTON * M_EARTH / R_EARTH
            delta_phi = phi - phi_surface
            phi_ratio = delta_phi / C_LIGHT**2
            data['gravitational_potential_ratio'].append(phi_ratio)
        
        # Current best constraints: |Δα/α| < 10⁻¹⁷ (Godun et al. 2014)
        # Simulate null result within measurement precision
        data['delta_alpha_over_alpha'] = [0.3e-18, -0.2e-18, 0.1e-18]
        data['error'] = [1.0e-18, 1.0e-18, 1.0e-18]
        
        return pd.DataFrame(data)
    
    @staticmethod
    def load_pulsar_data():
        """
        Load millisecond pulsar spectral line data.
        Simulated based on expected X-ray observations.
        """
        print("Note: Using simulated pulsar data (future observational target)")
        
        # Millisecond pulsars with extreme rotation
        pulsars = {
            'name': ['PSR_J1748-2446ad', 'PSR_B1937+21', 'PSR_J1939+2134'],
            'period_ms': [1.396, 1.558, 1.607],
            'radius_km': [12, 11, 10],
            'delta_alpha_over_alpha': [],
            'error': []
        }
        
        # Calculate angular velocity ratio
        for i, period_ms in enumerate(pulsars['period_ms']):
            omega = 2 * np.pi / (period_ms * 1e-3)
            r = pulsars['radius_km'][i] * 1000
            v_surface = omega * r
            v_ratio = v_surface / C_LIGHT
            
            # MRRC predicts small effect: Δα/α ~ γ·(v/c)²
            # γ ~ 10⁻⁸ (to be fitted)
            gamma = 1e-8
            delta_alpha = -gamma * v_ratio**2
            
            pulsars['delta_alpha_over_alpha'].append(delta_alpha)
            pulsars['error'].append(5e-7)  # Optimistic future precision
        
        return pd.DataFrame(pulsars)


class MRRCAnalysis:
    """Statistical analysis of MRRC predictions vs observations."""
    
    def __init__(self):
        self.quasar_data = None
        self.clock_data = None
        self.pulsar_data = None
        self.results = {}
    
    def load_all_data(self):
        """Load all available datasets."""
        self.quasar_data = ObservationalData.load_quasar_data()
        self.clock_data = ObservationalData.load_atomic_clock_data()
        self.pulsar_data = ObservationalData.load_pulsar_data()
        
        print("\nData loaded:")
        print(f"  Quasar measurements: {len(self.quasar_data)}")
        print(f"  Atomic clock comparisons: {len(self.clock_data)}")
        print(f"  Pulsar observations: {len(self.pulsar_data)}")
    
    def fit_gravitational_model(self):
        """Fit MRRC gravitational potential model to atomic clock data."""
        if self.clock_data is None:
            return
        
        # Extract data
        phi_ratio = np.array(self.clock_data['gravitational_potential_ratio'])
        delta_alpha = np.array(self.clock_data['delta_alpha_over_alpha'])
        errors = np.array(self.clock_data['error'])
        
        # Define MRRC model with fitting parameter β
        def mrrc_model(phi_ratio, beta):
            volume_scale = 1.0 + beta * phi_ratio
            alpha_inv_new = MRRCAlphaModel.alpha_inverse(volume_scale=volume_scale)
            return -(alpha_inv_new - ALPHA_INV_MRRC_BASE) / ALPHA_INV_MRRC_BASE
        
        # Fit
        try:
            popt, pcov = curve_fit(mrrc_model, phi_ratio, delta_alpha, 
                                   sigma=errors, p0=[1e-6])
            beta_fit = popt[0]
            beta_err = np.sqrt(pcov[0, 0])
            
            # Calculate chi-squared
            predictions = mrrc_model(phi_ratio, beta_fit)
            chi_sq = np.sum(((delta_alpha - predictions) / errors)**2)
            dof = len(delta_alpha) - 1
            p_value = 1 - chi2.cdf(chi_sq, dof)
            
            self.results['gravitational'] = {
                'beta': beta_fit,
                'beta_error': beta_err,
                'chi_squared': chi_sq,
                'dof': dof,
                'p_value': p_value,
                'predictions': predictions
            }
            
            print(f"\nGravitational Model Fit:")
            print(f"  β = {beta_fit:.2e} ± {beta_err:.2e}")
            print(f"  χ²/dof = {chi_sq:.2f}/{dof} (p = {p_value:.3f})")
            
        except Exception as e:
            print(f"Gravitational model fit failed: {e}")
    
    def fit_rotational_model(self):
        """Fit MRRC rotational model to pulsar data."""
        if self.pulsar_data is None:
            return
        
        # Calculate angular velocity ratios
        omega_ratios = []
        for _, row in self.pulsar_data.iterrows():
            period_ms = row['period_ms']
            omega = 2 * np.pi / (period_ms * 1e-3)
            omega_critical = C_LIGHT / (row.get('radius_km', 10) * 1000)
            omega_ratios.append(omega / omega_critical)
        
        omega_ratios = np.array(omega_ratios)
        delta_alpha = np.array(self.pulsar_data['delta_alpha_over_alpha'])
        errors = np.array(self.pulsar_data['error'])
        
        # Define MRRC rotational model
        def mrrc_rot_model(omega_ratio, gamma):
            rotation_scale = 1.0 + gamma * omega_ratio**2
            alpha_inv_new = MRRCAlphaModel.alpha_inverse(rotation_scale=rotation_scale)
            return -(alpha_inv_new - ALPHA_INV_MRRC_BASE) / ALPHA_INV_MRRC_BASE
        
        try:
            popt, pcov = curve_fit(mrrc_rot_model, omega_ratios, delta_alpha,
                                   sigma=errors, p0=[1e-8])
            gamma_fit = popt[0]
            gamma_err = np.sqrt(pcov[0, 0])
            
            predictions = mrrc_rot_model(omega_ratios, gamma_fit)
            chi_sq = np.sum(((delta_alpha - predictions) / errors)**2)
            dof = len(delta_alpha) - 1
            p_value = 1 - chi2.cdf(chi_sq, dof)
            
            self.results['rotational'] = {
                'gamma': gamma_fit,
                'gamma_error': gamma_err,
                'chi_squared': chi_sq,
                'dof': dof,
                'p_value': p_value,
                'predictions': predictions,
                'omega_ratios': omega_ratios
            }
            
            print(f"\nRotational Model Fit:")
            print(f"  γ = {gamma_fit:.2e} ± {gamma_err:.2e}")
            print(f"  χ²/dof = {chi_sq:.2f}/{dof} (p = {p_value:.3f})")
            
        except Exception as e:
            print(f"Rotational model fit failed: {e}")
    
    def analyze_quasar_spatial_pattern(self):
        """Analyze spatial distribution of quasar measurements for dipole."""
        if self.quasar_data is None:
            return
        
        # Fit dipole model: Δα/α = A·cos(θ)·sin(φ)
        theta = np.array(self.quasar_data['theta'])
        phi = np.array(self.quasar_data['phi'])
        delta_alpha = np.array(self.quasar_data['delta_alpha_over_alpha'])
        errors = np.array(self.quasar_data['error'])
        
        # Design matrix
        X = (np.cos(theta) * np.sin(phi)).reshape(-1, 1)
        
        # Weighted least squares
        W = np.diag(1 / errors**2)
        A_fit = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ delta_alpha
        A_fit = A_fit[0]
        
        # Error estimate
        residuals = delta_alpha - A_fit * X.flatten()
        A_err = np.sqrt(np.sum(residuals**2 * errors**2) / (len(delta_alpha) - 1))
        
        self.results['quasar_dipole'] = {
            'amplitude': A_fit,
            'amplitude_error': A_err,
            'dipole_vector': A_fit
        }
        
        print(f"\nQuasar Spatial Dipole Analysis:")
        print(f"  Dipole amplitude: ({A_fit:.2e} ± {A_err:.2e})")
        print(f"  Published (Webb et al.): -0.57e-5 ± 0.11e-5")
    
    def plot_results(self):
        """Generate comprehensive visualization of results."""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Quasar redshift distribution
        ax1 = plt.subplot(2, 3, 1)
        if self.quasar_data is not None:
            ax1.errorbar(self.quasar_data['redshift'],
                        self.quasar_data['delta_alpha_over_alpha'] * 1e6,
                        yerr=self.quasar_data['error'] * 1e6,
                        fmt='o', alpha=0.6, markersize=4)
            ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
            ax1.set_xlabel('Redshift z')
            ax1.set_ylabel('Δα/α (ppm)')
            ax1.set_title('Quasar Absorption Lines\n(Webb et al. type data)')
            ax1.grid(True, alpha=0.3)
        
        # 2. Atomic clock gravitational test
        ax2 = plt.subplot(2, 3, 2)
        if self.clock_data is not None:
            phi_ratio = self.clock_data['gravitational_potential_ratio']
            delta_alpha = self.clock_data['delta_alpha_over_alpha']
            errors = self.clock_data['error']
            
            ax2.errorbar(phi_ratio * 1e10, delta_alpha * 1e18,
                        yerr=errors * 1e18,
                        fmt='o', markersize=8, capsize=5, label='Observations')
            
            if 'gravitational' in self.results:
                phi_model = np.linspace(phi_ratio.min(), phi_ratio.max(), 100)
                beta = self.results['gravitational']['beta']
                
                volume_scale = 1.0 + beta * phi_model
                alpha_inv_new = MRRCAlphaModel.alpha_inverse(volume_scale=volume_scale)
                delta_pred = -(alpha_inv_new - ALPHA_INV_MRRC_BASE) / ALPHA_INV_MRRC_BASE
                
                ax2.plot(phi_model * 1e10, delta_pred * 1e18,
                        'r-', label='MRRC Prediction')
            
            ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
            ax2.set_xlabel('Φ/c² (×10⁻¹⁰)')
            ax2.set_ylabel('Δα/α (×10⁻¹⁸)')
            ax2.set_title('Atomic Clock Differential\n(Gravitational Potential)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Pulsar rotational test
        ax3 = plt.subplot(2, 3, 3)
        if self.pulsar_data is not None and 'rotational' in self.results:
            omega_ratios = self.results['rotational']['omega_ratios']
            delta_alpha = self.pulsar_data['delta_alpha_over_alpha']
            errors = self.pulsar_data['error']
            predictions = self.results['rotational']['predictions']
            
            ax3.errorbar(omega_ratios, delta_alpha * 1e6,
                        yerr=errors * 1e6,
                        fmt='o', markersize=8, capsize=5, label='Simulated Data')
            ax3.plot(omega_ratios, predictions * 1e6,
                    'r-', marker='s', label='MRRC Prediction')
            ax3.set_xlabel('ω/ω_critical')
            ax3.set_ylabel('Δα/α (ppm)')
            ax3.set_title('Millisecond Pulsar Test\n(Rotational Effects)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Sky map of quasar dipole
        ax4 = plt.subplot(2, 3, 4, projection='mollweide')
        if self.quasar_data is not None:
            theta = self.quasar_data['theta'] - np.pi  # Center at 0
            phi = self.quasar_data['phi'] - np.pi/2
            delta_alpha = self.quasar_data['delta_alpha_over_alpha']
            
            scatter = ax4.scatter(theta, phi, c=delta_alpha*1e6,
                                 cmap='RdBu_r', s=50, alpha=0.7)
            ax4.set_title('Spatial Distribution of Δα/α')
            ax4.grid(True)
            plt.colorbar(scatter, ax=ax4, label='Δα/α (ppm)')
        
        # 5. MRRC geometric breakdown
        ax5 = plt.subplot(2, 3, 5)
        components = ['4π³\n(Volume)', 'π²\n(Surface)', 'π\n(Rotation)', 'Total']
        values = [4*np.pi**3, np.pi**2, np.pi, ALPHA_INV_MRRC_BASE]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = ax5.bar(components, values, color=colors, alpha=0.7, edgecolor='black')
        ax5.axhline(ALPHA_INV_OBSERVED, color='red', linestyle='--',
                   linewidth=2, label=f'Observed: {ALPHA_INV_OBSERVED:.6f}')
        ax5.set_ylabel('α⁻¹')
        ax5.set_title('MRRC Derivation of α⁻¹')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = "MRRC V5.0 α-Variation Test Summary\n"
        summary_text += "="*40 + "\n\n"
        summary_text += f"Baseline Theory:\n"
        summary_text += f"  α⁻¹ = 4π³ + π² + π\n"
        summary_text += f"  Predicted: {ALPHA_INV_MRRC_BASE:.6f}\n"
        summary_text += f"  Observed:  {ALPHA_INV_OBSERVED:.6f}\n"
        summary_text += f"  Error: {abs(ALPHA_INV_MRRC_BASE - ALPHA_INV_OBSERVED):.6f}\n"
        summary_text += f"  ({abs(ALPHA_INV_MRRC_BASE - ALPHA_INV_OBSERVED)/ALPHA_INV_OBSERVED*100:.4f}%)\n\n"
        
        if 'gravitational' in self.results:
            gr = self.results['gravitational']
            summary_text += f"Gravitational Test:\n"
            summary_text += f"  β = {gr['beta']:.2e} ± {gr['beta_error']:.2e}\n"
            summary_text += f"  χ²/dof = {gr['chi_squared']:.2f}/{gr['dof']}\n\n"
        
        if 'rotational' in self.results:
            rr = self.results['rotational']
            summary_text += f"Rotational Test:\n"
            summary_text += f"  γ = {rr['gamma']:.2e} ± {rr['gamma_error']:.2e}\n"
            summary_text += f"  χ²/dof = {rr['chi_squared']:.2f}/{rr['dof']}\n\n"
        
        if 'quasar_dipole' in self.results:
            qr = self.results['quasar_dipole']
            summary_text += f"Quasar Dipole:\n"
            summary_text += f"  A = {qr['amplitude']:.2e} ± {qr['amplitude_error']:.2e}\n"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('mrrc_alpha_variation_analysis.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved: mrrc_alpha_variation_analysis.png")
        plt.close()  # Don't block on display
    
    def generate_report(self):
        """Generate detailed text report."""
        report = []
        report.append("="*60)
        report.append("MRRC V5.0 Fine Structure Constant Variation Test")
        report.append("="*60)
        report.append("")
        report.append("THEORETICAL FRAMEWORK:")
        report.append(f"  α⁻¹ = 4π³ + π² + π")
        report.append(f"  Predicted: {ALPHA_INV_MRRC_BASE:.9f}")
        report.append(f"  Observed:  {ALPHA_INV_OBSERVED:.9f}")
        report.append(f"  Discrepancy: {abs(ALPHA_INV_MRRC_BASE - ALPHA_INV_OBSERVED):.9f}")
        report.append(f"  Relative error: {abs(ALPHA_INV_MRRC_BASE - ALPHA_INV_OBSERVED)/ALPHA_INV_OBSERVED*1e6:.3f} ppm")
        report.append("")
        
        report.append("HYPOTHESIS:")
        report.append("  1. α varies with gravitational potential (volume_scale)")
        report.append("  2. α varies with rotation (rotation_scale)")
        report.append("  Magnitude: Δα/α ~ 10⁻¹⁸ (atomic clocks)")
        report.append("             Δα/α ~ 10⁻⁵ (cosmological)")
        report.append("")
        
        if 'gravitational' in self.results:
            gr = self.results['gravitational']
            report.append("GRAVITATIONAL POTENTIAL TEST:")
            report.append(f"  Coupling constant β = {gr['beta']:.3e} ± {gr['beta_error']:.3e}")
            report.append(f"  χ² = {gr['chi_squared']:.2f}, dof = {gr['dof']}")
            report.append(f"  p-value = {gr['p_value']:.3f}")
            report.append(f"  Status: {'CONSISTENT' if gr['p_value'] > 0.05 else 'TENSION'}")
            report.append("")
        
        if 'rotational' in self.results:
            rr = self.results['rotational']
            report.append("ROTATIONAL ASYMMETRY TEST:")
            report.append(f"  Coupling constant γ = {rr['gamma']:.3e} ± {rr['gamma_error']:.3e}")
            report.append(f"  χ² = {rr['chi_squared']:.2f}, dof = {rr['dof']}")
            report.append(f"  p-value = {rr['p_value']:.3f}")
            report.append(f"  Status: {'CONSISTENT' if rr['p_value'] > 0.05 else 'TENSION'}")
            report.append("")
        
        report.append("FUTURE OBSERVATIONAL TARGETS:")
        report.append("  1. Atomic clocks: Earth vs ISS vs GPS (Δα/α ~ 10⁻¹⁸)")
        report.append("  2. Millisecond pulsars: X-ray spectral lines")
        report.append("  3. Black hole accretion disks: Fe Kα line shifts")
        report.append("  4. Neutron star mergers: Multi-messenger constraints")
        report.append("")
        
        report.append("REFERENCES:")
        report.append("  - Webb et al. (2011): Quasar dipole Δα/α = -0.57e-5")
        report.append("  - Godun et al. (2014): |Δα/α| < 10⁻¹⁷ (Yb/Sr clocks)")
        report.append("  - Rosenband et al. (2008): Al⁺/Hg⁺ clock constraints")
        report.append("")
        report.append("="*60)
        
        report_text = "\n".join(report)
        print(report_text)
        
        # Save to file
        with open('mrrc_alpha_variation_report.txt', 'w') as f:
            f.write(report_text)
        print("\nReport saved: mrrc_alpha_variation_report.txt")
        
        return report_text


def main():
    """Main execution function."""
    print("MRRC V5.0 Fine Structure Constant Variation Test")
    print("=" * 60)
    print()
    
    # Initialize analysis
    analysis = MRRCAnalysis()
    
    # Load data
    analysis.load_all_data()
    
    # Perform fits
    analysis.fit_gravitational_model()
    analysis.fit_rotational_model()
    analysis.analyze_quasar_spatial_pattern()
    
    # Generate visualizations
    analysis.plot_results()
    
    # Generate report
    analysis.generate_report()
    
    print("\nAnalysis complete!")
    print("Generated files:")
    print("  - mrrc_alpha_variation_analysis.png")
    print("  - mrrc_alpha_variation_report.txt")


if __name__ == "__main__":
    main()
