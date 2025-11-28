import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from alpha_variation_data import ingestion

"""
Quick quasar dipole test
- Loads alpha_variation_data/alpha_variation_quasar_data.csv
- Fits dipole: Δα/α = A·cos(θ)·sin(φ)
- Outputs plot and console stats
"""

def load_data():
    override = os.getenv('QUASAR_CATALOG_PATH')
    df = ingestion.load_quasar_catalog(override) if override else ingestion.load_quasar_catalog()
    if (df.get('source') == 'simulated').all():
        raise RuntimeError('Quasar dataset is simulated. Set QUASAR_CATALOG_PATH to a real catalog CSV.')
    return df


def fit_dipole(df):
    theta = np.array(df['ra_deg']) * np.pi/180.0
    phi = (np.array(df['dec_deg']) + 90.0) * np.pi/180.0  # map to 0..π
    X = (np.cos(theta) * np.sin(phi)).reshape(-1, 1)
    y = np.array(df['delta_alpha_over_alpha'])
    w = 1.0 / np.clip(np.array(df['error']), 1e-12, None)
    W = np.diag(w**2)
    A = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
    return float(A[0])


def main():
    df = load_data()
    A = fit_dipole(df)
    print(f"Dipole amplitude A ≈ {A:.3e} (Webb ~ -5.7e-6)")
    plt.figure(figsize=(6,4))
    plt.errorbar(df['redshift'], df['delta_alpha_over_alpha']*1e6, yerr=df['error']*1e6, fmt='o', alpha=0.6)
    plt.axhline(0, color='k', ls='--', alpha=0.3)
    plt.xlabel('Redshift z')
    plt.ylabel('Δα/α (ppm)')
    plt.title('Quasar Δα/α vs z')
    plt.tight_layout()
    plt.savefig('quasar_dipole_plot.png', dpi=200)
    plt.close()
    print('Saved: quasar_dipole_plot.png')

if __name__ == '__main__':
    main()
