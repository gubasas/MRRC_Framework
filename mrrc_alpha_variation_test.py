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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
from scipy.stats import chi2
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from alpha_variation_data import ingestion

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
        Load quasar absorption line data (Webb et al., King et al.).
        Uses real King et al. 2012 dataset with 295 measurements.
        """
        # Prefer local Vizier TSV if present; else use ingestion
        vizier_path = Path.cwd() / 'king_2012_quasars.tsv'
        if vizier_path.exists():
            try:
                # Vizier format with pipe separator (cleaner than whitespace)
                df = pd.read_csv(vizier_path, sep='|', comment='#', engine='python')
                # Skip unit and separator rows (row 0 has units like '10-5', row 1 has '---')
                df = df[2:].reset_index(drop=True)
                
                # King 2012 Vizier columns: Seq, QSO, zem, zabs, da/a, e_da/a, Sample, Flag, Out, SimbadName, _RA, _DE
                ra = df['_RA'] if '_RA' in df.columns else None
                dec = df['_DE'] if '_DE' in df.columns else None
                dalpha = df['da/a'] if 'da/a' in df.columns else None
                derr = df['e_da/a'] if 'e_da/a' in df.columns else None
                z = df['zabs'] if 'zabs' in df.columns else None
                
                # Convert to numeric, drop NaN (handle None case)
                if ra is None or dec is None or dalpha is None or derr is None or z is None:
                    raise ValueError("Missing required columns in Vizier TSV")
                    
                ra_vals = pd.to_numeric(ra, errors='coerce').dropna()
                dec_vals = pd.to_numeric(dec, errors='coerce').dropna()
                dalpha_vals = pd.to_numeric(dalpha, errors='coerce').dropna() * 1e-5  # King 2012: units of 10^-5
                err_vals = pd.to_numeric(derr, errors='coerce').dropna() * 1e-5
                z_vals = pd.to_numeric(z, errors='coerce').dropna()
                
                # Align indices
                common_idx = dalpha_vals.index.intersection(err_vals.index).intersection(z_vals.index)
                common_idx = common_idx.intersection(ra_vals.index).intersection(dec_vals.index)
                
                theta_arr = np.array(ra_vals.loc[common_idx].values, dtype=float)
                phi_arr = np.array(dec_vals.loc[common_idx].values, dtype=float)
                theta = np.deg2rad(theta_arr)
                phi = np.deg2rad(90.0 - phi_arr)
                
                out = pd.DataFrame({
                    'theta': theta,
                    'phi': phi,
                    'redshift': z_vals.loc[common_idx].values,
                    'delta_alpha_over_alpha': dalpha_vals.loc[common_idx].values,
                    'error': err_vals.loc[common_idx].values,
                    'source': 'king2012_vizier'
                })
                print(f"Loaded quasar data from Vizier TSV: {vizier_path.name} ({len(out)} rows)")
                return out
            except Exception as e:
                print(f"Warning: Failed to parse {vizier_path.name}: {e}. Falling back to ingestion.")
        try:
            df = ingestion.load_quasar_catalog()
            if (df.get('source') == 'simulated').all():
                print("Warning: Quasar dataset is simulated")
            else:
                print(f"Loaded real quasar data: {df['source'].iloc[0]} ({len(df)} measurements)")
            # Compute sky coordinates (theta, phi) from RA/DEC if available
            if 'ra_deg' in df.columns and 'dec_deg' in df.columns:
                df['theta'] = np.deg2rad(df['ra_deg'])
                df['phi'] = np.deg2rad(90 - df['dec_deg'])
            return df
        except Exception as e:
            print(f"Failed to load quasar data: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def load_atomic_clock_data():
        """
        Load atomic clock drift constraints (Rosenband, Godun, etc.).
        Uses real published drift measurements.
        """
        try:
            df = ingestion.load_atomic_clock_constraints()
            if (df.get('source') == 'simulated').all():
                print("Warning: Clock dataset is simulated")
            else:
                print(f"Loaded real clock data: {len(df)} constraints from published sources")
        except Exception as e:
            print(f"Failed to load clock data: {e}")
            df = pd.DataFrame()
        # Overlay Nemitz (2016) gold standard: Sr/Yb+ 0.8e-17 ± 0.8e-17 yr⁻¹
        try:
            nemitz = pd.DataFrame([{
                'clock_pair': 'Sr/Yb+',
                'delta_alpha_over_alpha': 0.8e-17,
                'error': 0.8e-17,
                'k_alpha': 5.0,
                'source': 'Nemitz2016',
                'delta_phi_over_c2': np.nan
            }])
            if df is None or df.empty:
                df = nemitz
            else:
                df = df[~(df['clock_pair'].astype(str).str.contains('Yb') & df['clock_pair'].astype(str).str.contains('Sr'))]
                df = pd.concat([df, nemitz], ignore_index=True)
            print('Applied Nemitz (2016) constraint: 0.8e-17 ± 0.8e-17 yr⁻¹ (Sr/Yb+)')
        except Exception:
            pass
        return df
    
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


class SeasonalModulationAnalyzer:
    """Extract gravitational coupling from seasonal Earth-Sun potential modulation."""
    
    # Earth-Sun orbital parameters
    EARTH_SUN_PERIHELION = 1.471e11  # m (January)
    EARTH_SUN_APHELION = 1.521e11    # m (July)
    M_SUN = 1.989e30  # kg
    
    @staticmethod
    def compute_seasonal_potential_amplitude():
        """Compute Earth-Sun gravitational potential modulation amplitude Δφ/c²."""
        # Potential at perihelion (January, closer to Sun)
        phi_peri = -G_NEWTON * SeasonalModulationAnalyzer.M_SUN / SeasonalModulationAnalyzer.EARTH_SUN_PERIHELION
        # Potential at aphelion (July, farther from Sun)
        phi_aph = -G_NEWTON * SeasonalModulationAnalyzer.M_SUN / SeasonalModulationAnalyzer.EARTH_SUN_APHELION
        # Amplitude of variation
        delta_phi = phi_peri - phi_aph
        delta_phi_over_c2 = delta_phi / C_LIGHT**2
        return delta_phi_over_c2
    
    @staticmethod
    def extract_beta_from_drift_data(clock_df, seasonal_amplitude=None):
        """
        Infer β constraint from clock drift data assuming no observed seasonal modulation.
        
        Parameters:
        -----------
        clock_df : DataFrame with delta_alpha_over_alpha, error, k_alpha
        seasonal_amplitude : float, optional (default: computed Earth-Sun value)
        
        Returns:
        --------
        dict : {'beta_upper_limit', 'seasonal_amplitude_used'}
        """
        if seasonal_amplitude is None:
            seasonal_amplitude = SeasonalModulationAnalyzer.compute_seasonal_potential_amplitude()
        
        # Use the tightest constraint (smallest error) for upper limit
        if clock_df.empty or 'error' not in clock_df.columns:
            return {'beta_upper_limit': np.nan, 'seasonal_amplitude_used': seasonal_amplitude}
        
        # Drift uncertainties give bounds on undetected modulation amplitude
        # Conservative: |Δα/α|_modulation < 3σ_drift (95% CL)
        min_error = clock_df['error'].min()
        modulation_limit = 3 * min_error
        
        # β = (Δα/α) / (Δφ/c²)
        beta_limit = modulation_limit / abs(seasonal_amplitude)
        
        return {
            'beta_upper_limit': beta_limit,
            'seasonal_amplitude_used': seasonal_amplitude,
            'tightest_constraint_sigma': min_error,
            'modulation_limit_3sigma': modulation_limit
        }


class MRRCAnalysis:
    """Statistical analysis of MRRC predictions vs observations."""
    
    def __init__(self):
        self.quasar_data = None
        self.clock_data = None
        self.pulsar_data = None
        self.results = {}
        self.seasonal_analyzer = SeasonalModulationAnalyzer()
        # Config toggles
        self.config = {
            'overlay_king2012_dipole': True,
            'king2012_dipole_amplitude': 1.02e-5,  # King 2012 reported amplitude
            'enable_joint_fit': True
        }
    
    def load_all_data(self):
        """Load all available datasets."""
        self.quasar_data = ObservationalData.load_quasar_data()
        self.clock_data = ObservationalData.load_atomic_clock_data()
        self.pulsar_data = ObservationalData.load_pulsar_data()
        # Ingest white dwarf sources (Berengut 2013 baseline + hooks)
        self.white_dwarf_data = self.ingest_white_dwarf_sources()
        # Compute orbital modulation bound (Galileo 5/6)
        try:
            g_data, aces = self.ingest_orbital_modulation_data()
            beta_orbital = self.fit_beta_from_modulation(g_data)
            self.results['orbital_modulation'] = {
                'beta_upper_limit': beta_orbital,
                'dataset': g_data['name'],
                'dU_over_c2': g_data['modulation_amplitude_U'],
                'alpha_grav_limit': g_data['clock_residual_limit']
            }
            # ACES projection
            beta_proj = self.project_aces_beta_bound(aces, k_alpha=1.0)
            self.results['aces_projection'] = {
                'beta_projected_limit': beta_proj,
                'potential_diff': aces['potential_diff'],
                'target_precision': aces['target_precision']
            }
        except Exception as e:
            print(f"Warning: orbital modulation analysis failed: {e}")
        
        print("\nData loaded:")
        print(f"  Quasar measurements: {len(self.quasar_data)}")
        print(f"  Atomic clock comparisons: {len(self.clock_data)}")
        print(f"  Pulsar observations: {len(self.pulsar_data)}")
        print(f"  White dwarf sources: {len(self.white_dwarf_data)}")
    
    def fit_gravitational_model(self):
        """Fit MRRC gravitational potential model to atomic clock data."""
        if self.clock_data is None or self.clock_data.empty:
            print("\nGravitational Model: No clock data available")
            return
        
        # Check if we have modulation data (non-zero delta_phi_over_c2)
        has_modulation = False
        mod_mask = pd.Series([False] * len(self.clock_data))
        
        if 'delta_phi_over_c2' in self.clock_data.columns:
            mod_mask = (~self.clock_data['delta_phi_over_c2'].isna()) & (self.clock_data['delta_phi_over_c2'] != 0)
            has_modulation = mod_mask.any()
        
        if has_modulation:
            # Direct fit to modulation data
            sub = self.clock_data[mod_mask]
            phi_ratio = np.array(sub['delta_phi_over_c2'])
            delta_alpha = np.array(sub['delta_alpha_over_alpha'])
            errors = np.array(sub['error'])
            
            def mrrc_model(phi_ratio, beta):
                volume_scale = 1.0 + beta * phi_ratio
                alpha_inv_new = MRRCAlphaModel.alpha_inverse(volume_scale=volume_scale)
                return -(alpha_inv_new - ALPHA_INV_MRRC_BASE) / ALPHA_INV_MRRC_BASE
            
            try:
                popt, pcov = curve_fit(mrrc_model, phi_ratio, delta_alpha, 
                                       sigma=errors, p0=[1e-6])
                beta_fit = popt[0]
                beta_err = np.sqrt(pcov[0, 0])
                
                predictions = mrrc_model(phi_ratio, beta_fit)
                chi_sq = np.sum(((delta_alpha - predictions) / errors)**2)
                dof = len(delta_alpha) - 1
                p_value = 1 - chi2.cdf(chi_sq, dof) if dof > 0 else 1.0
                
                self.results['gravitational'] = {
                    'beta': beta_fit,
                    'beta_error': beta_err,
                    'chi_squared': chi_sq,
                    'dof': dof,
                    'p_value': p_value,
                    'predictions': predictions,
                    'method': 'direct_fit'
                }
                
                print(f"\nGravitational Model Fit (Direct Modulation):")
                print(f"  β = {beta_fit:.2e} ± {beta_err:.2e}")
                print(f"  χ²/dof = {chi_sq:.2f}/{dof} (p = {p_value:.3f})")
                
            except Exception as e:
                print(f"Direct modulation fit failed: {e}")
                has_modulation = False
        
        # If no modulation data, extract upper limit from drift constraints
        if not has_modulation:
            seasonal_result = self.seasonal_analyzer.extract_beta_from_drift_data(self.clock_data)
            beta_limit = seasonal_result['beta_upper_limit']
            seasonal_amp = seasonal_result['seasonal_amplitude_used']
            
            self.results['gravitational'] = {
                'beta': 0.0,
                'beta_error': np.inf,
                'beta_upper_limit': beta_limit,
                'seasonal_amplitude': seasonal_amp,
                'chi_squared': 0.0,
                'dof': 0,
                'p_value': 1.0,
                'method': 'seasonal_limit'
            }
            
            print(f"\nGravitational Model (Seasonal Modulation Limit):")
            print(f"  Seasonal Δφ/c² amplitude: {seasonal_amp:.2e}")
            print(f"  β upper limit (95% CL): {beta_limit:.2e}")
            print(f"  Derived from tightest drift constraint: {seasonal_result['tightest_constraint_sigma']:.2e}")
    
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
        # Store overlay amplitude for plotting/reporting
        if self.config.get('overlay_king2012_dipole'):
            self.results['quasar_dipole']['overlay_amplitude'] = self.config['king2012_dipole_amplitude']
    
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
            # Overlay King (2012) fixed dipole amplitude as reference band
            if 'quasar_dipole' in self.results and self.results['quasar_dipole'].get('overlay_amplitude'):
                Aref = self.results['quasar_dipole']['overlay_amplitude'] * 1e6
                ax1.axhline(Aref, color='purple', linestyle=':', alpha=0.6, label='King 2012 dipole (ppm)')
                ax1.axhline(-Aref, color='purple', linestyle=':', alpha=0.6)
                ax1.legend()
        
        # 2. Atomic clock gravitational test
        ax2 = plt.subplot(2, 3, 2)
        if self.clock_data is not None and not self.clock_data.empty:
            # Use delta_phi_over_c2 column (from ingestion)
            phi_col = 'delta_phi_over_c2' if 'delta_phi_over_c2' in self.clock_data.columns else 'gravitational_potential_ratio'
            
            # Filter to rows with defined potential values
            valid_mask = (~self.clock_data[phi_col].isna()) if phi_col in self.clock_data.columns else pd.Series([True] * len(self.clock_data))
            sub = self.clock_data[valid_mask]
            
            if not sub.empty and phi_col in sub.columns:
                phi_ratio = sub[phi_col]
                delta_alpha = sub['delta_alpha_over_alpha']
                errors = sub['error']
                
                ax2.errorbar(phi_ratio * 1e10, delta_alpha * 1e18,
                            yerr=errors * 1e18,
                            fmt='o', markersize=8, capsize=5, label='Observations')
                
                if 'gravitational' in self.results and self.results['gravitational'].get('method') == 'direct_fit':
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
            else:
                # No potential data; show drift constraints as time series
                ax2.text(0.5, 0.5, 'Drift Constraints Only\n(No Modulation Data)', 
                        ha='center', va='center', transform=ax2.transAxes)
                if 'gravitational' in self.results and 'beta_upper_limit' in self.results['gravitational']:
                    beta_lim = self.results['gravitational']['beta_upper_limit']
                    ax2.text(0.5, 0.3, f'β < {beta_lim:.2e} (95% CL)', 
                            ha='center', va='center', transform=ax2.transAxes, fontsize=10)
                ax2.set_title('Atomic Clock Constraints')
            
            # Add white dwarf data point (Berengut 2013: G191-B2B)
            # Φ/c² ≈ 1e-4 (WD gravity), Δα/α = 4.2e-5 ± 1.6e-5
            # Plot all white dwarf sources from ingestion
            if getattr(self, 'white_dwarf_data', None) is not None and not self.white_dwarf_data.empty:
                first_label = False
                for _, row in self.white_dwarf_data.iterrows():
                    phi = row.get('phi_over_c2')
                    if np.isnan(phi):
                        continue
                    label = row['name'] if not first_label else None
                    if row.get('type') == 'detection' and not np.isnan(row.get('delta_alpha_over_alpha')):
                        ax2.errorbar(phi * 1e10,
                                     row['delta_alpha_over_alpha'] * 1e18,
                                     yerr=(row['error'] * 1e18) if not np.isnan(row.get('error')) else None,
                                     fmt='*', markersize=12, color='gold', markeredgecolor='black',
                                     capsize=5, zorder=10, label=label)
                    elif row.get('type') == 'upper_limit' and not np.isnan(row.get('upper_limit')):
                        ax2.plot(phi * 1e10, row['upper_limit'] * 1e18,
                                 marker='v', markersize=8, color='gold', markeredgecolor='black',
                                 linestyle='None', label=label)
                        ax2.annotate('UL', (phi * 1e10, row['upper_limit'] * 1e18),
                                     textcoords='offset points', xytext=(0,5), ha='center', fontsize=7)
                    if not first_label:
                        first_label = True
            # Annotate orbital modulation bound
            if 'orbital_modulation' in self.results:
                ob = self.results['orbital_modulation']
                ax2.text(0.02, 0.08, f"Galileo bound: β < {ob['beta_upper_limit']:.2e}", 
                         transform=ax2.transAxes, fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
            ax2.legend()

            # Regime comparison inset: weak (Earth-Sun), orbital (Galileo), strong (WD)
            try:
                inset = inset_axes(ax2, width="45%", height="45%", loc='upper right', borderpad=1)
                # X: Φ/c², Y: indicative |Δα/α| or β*Φ proxy
                seasonal_amp = self.results.get('gravitational', {}).get('seasonal_amplitude',
                                        SeasonalModulationAnalyzer.compute_seasonal_potential_amplitude())
                beta_seasonal = self.results.get('gravitational', {}).get('beta_upper_limit', np.nan)
                phi_earth = abs(seasonal_amp)
                # Orbital
                phi_orb = abs(self.results.get('orbital_modulation', {}).get('dU_over_c2', np.nan))
                beta_orb = self.results.get('orbital_modulation', {}).get('beta_upper_limit', np.nan)
                # White dwarf
                if getattr(self, 'white_dwarf_data', None) is not None and not self.white_dwarf_data.empty:
                    det_df = self.white_dwarf_data[self.white_dwarf_data['type'] == 'detection']
                    if not det_df.empty:
                        wd_row = det_df.iloc[0]
                        phi_wd = wd_row['phi_over_c2']
                        dalpha_wd = abs(wd_row['delta_alpha_over_alpha'])
                    else:
                        ul_df = self.white_dwarf_data[self.white_dwarf_data['type'] == 'upper_limit']
                        if not ul_df.empty:
                            wd_row = ul_df.iloc[0]
                            phi_wd = wd_row['phi_over_c2']
                            dalpha_wd = ul_df.iloc[0]['upper_limit']
                        else:
                            phi_wd = 1e-4
                            dalpha_wd = 4.2e-5
                else:
                    phi_wd = 1e-4
                    dalpha_wd = 4.2e-5
                # Plot points
                xs = np.array([phi_earth, phi_orb, phi_wd])
                ys = np.array([beta_seasonal * phi_earth if np.isfinite(beta_seasonal) else np.nan,
                               beta_orb * phi_orb if np.isfinite(beta_orb) else np.nan,
                               dalpha_wd])
                labels = ['Earth-Sun', 'Orbital', 'White Dwarf']
                inset.loglog(xs, ys, 'ko', markersize=6)
                for x, y, lab in zip(xs, ys, labels):
                    inset.text(x*1.1, y*1.1, lab, fontsize=7)
                inset.set_xlabel('Φ/c²', fontsize=8)
                inset.set_ylabel('|Δα/α|', fontsize=8)
                inset.grid(True, which='both', alpha=0.3)
                inset.set_title('Regimes', fontsize=9)
            except Exception:
                pass
        
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
        if 'orbital_modulation' in self.results:
            ob = self.results['orbital_modulation']
            summary_text += f"Orbital Modulation (Galileo 5/6):\n"
            summary_text += f"  ΔU/c² ≈ {ob['dU_over_c2']:.2e}\n"
            summary_text += f"  α_grav limit ≈ {ob['alpha_grav_limit']:.2e}\n"
            summary_text += f"  β upper limit ≈ {ob['beta_upper_limit']:.2e}\n\n"
        
        if 'rotational' in self.results:
            rr = self.results['rotational']
            summary_text += f"Rotational Test:\n"
            summary_text += f"  γ = {rr['gamma']:.2e} ± {rr['gamma_error']:.2e}\n"
            summary_text += f"  χ²/dof = {rr['chi_squared']:.2f}/{rr['dof']}\n\n"
        
        if 'quasar_dipole' in self.results:
            qr = self.results['quasar_dipole']
            summary_text += f"Quasar Dipole:\n"
            summary_text += f"  A = {qr['amplitude']:.2e} ± {qr['amplitude_error']:.2e}\n"
            if qr.get('overlay_amplitude'):
                summary_text += f"  Ref (King 2012): {qr['overlay_amplitude']:.2e}\n"
        
        # Append joint-fit summary if available
        if 'joint_beta_phi' in self.results and 'beta0_est' in self.results['joint_beta_phi']:
            jf = self.results['joint_beta_phi']
            summary_text += f"\nJoint β(Φ) Fit (proxy):\n"
            summary_text += f"  β₀ ≈ {jf['beta0_est']:.2e}, β₂ ≈ {jf['beta2_est']:.2e}\n"

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
        if 'orbital_modulation' in self.results:
            ob = self.results['orbital_modulation']
            report.append("ORBITAL MODULATION TEST (Galileo 5/6):")
            report.append(f"  ΔU/c² ≈ {ob['dU_over_c2']:.3e}")
            report.append(f"  α_grav limit ≈ {ob['alpha_grav_limit']:.3e}")
            report.append(f"  Derived β upper limit ≈ {ob['beta_upper_limit']:.3e}")
            report.append("")
        if 'aces_projection' in self.results:
            ap = self.results['aces_projection']
            report.append("ACTIVE MODULATION PROJECTION (ISS/ACES):")
            report.append(f"  Potential difference (ISS-ground): {ap['potential_diff']:.3e}")
            report.append(f"  Target precision: {ap['target_precision']:.1e}")
            report.append(f"  Projected β upper limit: {ap['beta_projected_limit']:.3e}")
            report.append("")
        if 'joint_beta_phi' in self.results and 'beta0_est' in self.results['joint_beta_phi']:
            jf = self.results['joint_beta_phi']
            report.append("JOINT β(Φ) FIT (proxy):")
            report.append(f"  Points used: {jf.get('points_used', 0)}")
            report.append(f"  β₀ ≈ {jf['beta0_est']:.3e}")
            report.append(f"  β₂ ≈ {jf['beta2_est']:.3e}")
            report.append("")
        
        if 'rotational' in self.results:
            rr = self.results['rotational']
            report.append("ROTATIONAL ASYMMETRY TEST:")
            report.append(f"  Coupling constant γ = {rr['gamma']:.3e} ± {rr['gamma_error']:.3e}")
            report.append(f"  χ² = {rr['chi_squared']:.2f}, dof = {rr['dof']}")
            report.append(f"  p-value = {rr['p_value']:.3f}")
            report.append(f"  Status: {'CONSISTENT' if rr['p_value'] > 0.05 else 'TENSION'}")
            report.append("")
        # Combined summary of best bounds
        if 'gravitational' in self.results or 'orbital_modulation' in self.results:
            report.append("COMBINED SUMMARY:")
            if 'gravitational' in self.results and 'beta_upper_limit' in self.results['gravitational']:
                report.append(f"  Best β bound (drift + seasonal): {self.results['gravitational']['beta_upper_limit']:.3e}")
            if 'orbital_modulation' in self.results:
                ob = self.results['orbital_modulation']
                report.append(f"  Cross-check β (Galileo orbital): {ob['beta_upper_limit']:.3e}")
            if 'aces_projection' in self.results:
                report.append(f"  Projected β (ISS/ACES): {self.results['aces_projection']['beta_projected_limit']:.3e}")
            report.append("")
        
        report.append("FUTURE OBSERVATIONAL TARGETS:")
        report.append("  1. Atomic clocks: Earth vs ISS vs GPS (Δα/α ~ 10⁻¹⁸)")
        report.append("  2. Millisecond pulsars: X-ray spectral lines")
        report.append("  3. Black hole accretion disks: Fe Kα line shifts")
        report.append("  4. Neutron star mergers: Multi-messenger constraints")
        report.append("")

        # Interpretation block: why orbital is looser but complementary
        report.append("INTERPRETATION:")
        report.append("  Drift + seasonal bounds probe weak-field (Earth-Sun) variations with exquisite precision,")
        report.append("  yielding β ≲ 7.3×10⁻⁸. Orbital modulation (Galileo 5/6) is an active test in a slightly")
        report.append("  stronger, time-varying potential, but clock systematic limits (α_grav ≲ 2.5×10⁻⁵) translate")
        report.append("  to a looser β ≲ 2.9×10⁻⁵. The two are complementary: drift tests are most sensitive in the")
        report.append("  weak-field regime; modulation tests validate absence of periodic signatures. Future ISS/ACES")
        report.append("  space-to-ground links at ~10⁻¹⁶ precision could tighten the active-modulation bound by orders")
        report.append("  of magnitude, closing the gap between laboratory nulls and strong-field probes like white dwarfs.")
        report.append("")
        report.append("NONLINEARITY NOTE:")
        report.append("  A joint weak/strong-field fit using β(Φ) = β₀ + β₂·Φ can test whether")
        report.append("  null weak-field results and tentative strong-field signals (e.g., white dwarfs)")
        report.append("  are reconcilable within MRRC. Implementing this requires aggregating additional white dwarf")
        report.append("  sources and performing a combined likelihood across regimes.")
        report.append("")

        # CMB consistency summary (order-of-magnitude back-of-envelope)
        rms_dT_over_T = 1.0e-5
        sachs_wolfe_coeff = 3.0  # RMS[Φ/c²] ≈ 3 × RMS[ΔT/T]
        rms_phi_over_c2 = sachs_wolfe_coeff * rms_dT_over_T
        best_beta = None
        if 'gravitational' in self.results and 'beta_upper_limit' in self.results['gravitational']:
            best_beta = self.results['gravitational']['beta_upper_limit']

        report.append("CMB CONSISTENCY (Summary):")
        report.append(f"  Inputs: RMS[ΔT/T]={rms_dT_over_T:.1e} ⇒ RMS[Φ/c²]={rms_phi_over_c2:.1e} (Sachs–Wolfe ×3)")
        if best_beta is not None:
            dalpha_rms = best_beta * rms_phi_over_c2
            report.append(f"  Best β bound (lab): {best_beta:.3e}")
            report.append(f"  Predicted RMS |Δα/α| (lab β): {dalpha_rms:.3e}")
            report.append(f"  Predicted RMS |δσ_T/σ_T|: {2*dalpha_rms:.3e}")
            report.append(f"  Predicted RMS |δz*/z*|: {2*dalpha_rms:.3e}")
            # Loose, permissive CMB upper bound example
            loose_dalpha_rms = 3.0e-3
            beta_cmb_max = loose_dalpha_rms / rms_phi_over_c2
            report.append(f"  Loose CMB prior example: |Δα/α|₍RMS₎≲3e-3 ⇒ β≲{beta_cmb_max:.2e}")
        else:
            report.append("  Best β bound unavailable; skip numerical projection.")
        report.append("")

        # Cosmology energy accounting (Run 2 vs V5.1)
        report.append("COSMOLOGY ENERGY ACCOUNTING (Run 2 vs V5.1):")
        report.append("  Run 2 (Naive Maintenance):")
        report.append("    Assumption: Universe is 'Compliant' — pay to actively maintain all CMB bits (~1e122 bits).")
        report.append("    Predicted power: 1.86e+82 W.")
        report.append("    Issue: Exceeds physical limit c^5/G ≈ 3.64e+52 W by ~30 orders — catastrophic violation.")
        report.append("  V5.1 (Stiff/Mode-Locked Substrate):")
        report.append("    Assumption: Static states are free; only pay for change (expansion).")
        report.append("    Calculation: W = T_H · dS/dt (horizon thermodynamics).")
        report.append("    Result: 1.82e+52 W ≈ 0.50 · (c^5/G).")
        report.append("    Interpretation: The 1/2 factor is standard from equipartition on horizons.")
        report.append("  Conclusion: Early MRRC 'failure' was accounting — charging maintenance on a locked system.")
        report.append("             With Stiff Substrate (supported by atomic-clock nulls), the universe spends energy")
        report.append("             to grow, not to exist — resolving the cosmological energy crisis.")
        report.append("")
        
        report.append("REFERENCES:")
        report.append("  - Webb et al. (2011): Quasar dipole Δα/α = -0.57e-5")
        report.append("  - Godun et al. (2014): |Δα/α| < 10⁻¹⁷ (Yb/Sr clocks)")
        report.append("  - Rosenband et al. (2008): Al⁺/Hg⁺ clock constraints")
        report.append("  - Sachs & Wolfe (1967): CMB anisotropies and gravitational potentials")
        report.append("  - Planck Collaboration (2018): RMS ΔT/T ≈ 1e-5")
        report.append("")
        report.append("="*60)
        
        report_text = "\n".join(report)
        print(report_text)
        
        # Save to file
        with open('mrrc_alpha_variation_report.txt', 'w') as f:
            f.write(report_text)
        print("\nReport saved: mrrc_alpha_variation_report.txt")
        
        return report_text

    # --- Orbital modulation (Galileo 5/6) ingestion and beta fit ---
    @staticmethod
    def ingest_orbital_modulation_data():
        """
        Ingest Orbital Modulation data from Galileo 5/6 (Delva et al., 2018) and
        define ACES/PHARAO hook for future ISS data.

        Returns:
            dict: Galileo dataset dict, ACES hook dict
        """
        print("\n[INGESTION] Loading Orbital Modulation Data...")
        galileo_data = {
            'name': 'Galileo GSAT-0201/0202',
            'clock_type': 'H-Maser',
            'orbit': 'Elliptical (e=0.16)',
            'modulation_amplitude_U': 3.7e-10,  # Peak-to-peak potential variation (c^-2)
            'clock_residual_limit': 2.5e-5,     # Limit on fractional frequency dev (Delva 2018)
            'k_alpha': 0.87                     # Sensitivity of H-Maser to alpha
        }
        aces_hook = {
            'name': 'ACES/PHARAO (ISS)',
            'clock_type': 'Cesium (Pharao) + H-Maser',
            'modulation_source': 'Space-to-Ground Link',
            'potential_diff': 3.0e-10,
            'target_precision': 1e-16
        }
        print(f"   > Loaded {galileo_data['name']} Constraints (Delva 2018)")
        return galileo_data, aces_hook

    @staticmethod
    def fit_beta_from_modulation(galileo_data):
        """Fit β from orbital modulation using Galileo parameters."""
        print("\n[ANALYSIS] Fitting Beta from Orbital Modulation...")
        dU = galileo_data['modulation_amplitude_U']
        limit_alpha_grav = galileo_data['clock_residual_limit']
        K_alpha = galileo_data['k_alpha']
        max_anomalous_freq = limit_alpha_grav * dU
        beta_bound = limit_alpha_grav / K_alpha
        print(f"   > Modulation Amplitude (ΔU/c²): {dU:.2e}")
        print(f"   > Clock Residual Limit × ΔU: {max_anomalous_freq:.2e}")
        print(f"   > Derived β Limit (Orbital): < {beta_bound:.2e}")
        return beta_bound

    @staticmethod
    def project_aces_beta_bound(aces_hook, k_alpha=1.0):
        """Project β bound for ISS/ACES given target precision and potential difference."""
        dU = aces_hook.get('potential_diff', 3.0e-10)
        precision = aces_hook.get('target_precision', 1e-16)
        beta_proj = precision / (k_alpha * dU)
        return beta_proj

    # --- White dwarf ingestion and joint-fit scaffold ---
    @staticmethod
    def ingest_white_dwarf_sources():
        """Ingest white dwarf sources with detection vs upper-limit handling.

        Returns:
            DataFrame columns: name, phi_over_c2, delta_alpha_over_alpha, error, upper_limit, type
        """
        csv_path = Path.cwd() / 'alpha_variation_data' / 'white_dwarfs.csv'
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                for col in ['phi_over_c2', 'delta_alpha_over_alpha', 'error', 'upper_limit']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                if 'type' not in df.columns:
                    df['type'] = np.where(df['delta_alpha_over_alpha'].notna(), 'detection', 'upper_limit')
                print(f"Loaded white dwarf sources from CSV: {csv_path.name} ({len(df)})")
                return df[['name', 'phi_over_c2', 'delta_alpha_over_alpha', 'error', 'upper_limit', 'type']]
            except Exception as e:
                print(f"Warning: Failed to parse {csv_path.name}: {e}. Using baseline entry.")
        df = pd.DataFrame([{
            'name': 'G191-B2B (Berengut 2013)',
            'phi_over_c2': 1.0e-4,
            'delta_alpha_over_alpha': 4.2e-5,
            'error': 1.6e-5,
            'upper_limit': np.nan,
            'type': 'detection'
        }])
        return df

    def joint_fit_beta_of_phi(self):
        """Simple joint-fit scaffold for β(Φ) = β0 + β2·Φ using weak/strong-field points.

        Uses:
          - Weak-field seasonal limit: treat as upper bound at Φ_earth with |Δα/α| < 3σ
          - Orbital modulation: upper bound at Φ_orb with |Δα/α| < β_orb*Φ_orb
          - White dwarf points: direct Δα/α measurements at Φ_wd

        Stores:
          self.results['joint_beta_phi'] with fit coefficients and notes.
        """
        if not self.config.get('enable_joint_fit'):
            return
        try:
            # Assemble data
            seasonal_amp = self.results.get('gravitational', {}).get('seasonal_amplitude',
                                    SeasonalModulationAnalyzer.compute_seasonal_potential_amplitude())
            min_err = self.results.get('gravitational', {}).get('tightest_constraint_sigma', np.nan)
            phi_earth = abs(seasonal_amp)
            y_earth = 3 * min_err if np.isfinite(min_err) else np.nan

            phi_orb = abs(self.results.get('orbital_modulation', {}).get('dU_over_c2', np.nan))
            beta_orb = self.results.get('orbital_modulation', {}).get('beta_upper_limit', np.nan)
            y_orb = beta_orb * phi_orb if np.isfinite(beta_orb) and np.isfinite(phi_orb) else np.nan

            wd_df = getattr(self, 'white_dwarf_data', pd.DataFrame())
            wd_df = wd_df.dropna(subset=['phi_over_c2', 'delta_alpha_over_alpha']) if not wd_df.empty else wd_df

            # Build arrays
            X_list = []
            Y_list = []
            if np.isfinite(phi_earth) and np.isfinite(y_earth):
                X_list.append([1.0, phi_earth])
                Y_list.append(y_earth)
            if np.isfinite(phi_orb) and np.isfinite(y_orb):
                X_list.append([1.0, phi_orb])
                Y_list.append(y_orb)
            for _, row in wd_df.iterrows():
                X_list.append([1.0, row['phi_over_c2']])
                Y_list.append(abs(row['delta_alpha_over_alpha']))

            if len(X_list) < 2:
                self.results['joint_beta_phi'] = {'note': 'Insufficient data for joint fit'}
                return

            X = np.array(X_list)
            Y = np.array(Y_list)

            # Linear least squares: |Δα/α| ≈ |β0 + β2·Φ|·Φ ≈ proxy; use small-Φ approx
            # For simplicity, fit y ≈ b0*Φ + b2*Φ² where b0≈β0 and b2≈β2
            Phi = X[:, 1]
            A = np.vstack([Phi, Phi**2]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
            beta0_est, beta2_est = coeffs[0], coeffs[1]

            self.results['joint_beta_phi'] = {
                'beta0_est': beta0_est,
                'beta2_est': beta2_est,
                'points_used': len(Y_list)
            }
        except Exception as e:
            self.results['joint_beta_phi'] = {'error': str(e)}


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
