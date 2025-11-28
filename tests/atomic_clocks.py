import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from alpha_variation_data import ingestion

"""
Atomic clock gravitational potential test
- Loads alpha_variation_data/alpha_variation_clock_data.csv
- Fits Δα/α = f(Φ/c²) via linear MRRC coupling β
"""

def load_data():
    override = os.getenv('CLOCK_CATALOG_PATH')
    df = ingestion.load_atomic_clock_constraints(override) if override else ingestion.load_atomic_clock_constraints()
    if (df.get('source') == 'simulated').all():
        raise RuntimeError('Clock dataset is simulated. Set CLOCK_CATALOG_PATH to a real constraints CSV.')
    return df


def fit_beta(df):
    x = np.array(df['delta_phi_over_c2'] if 'delta_phi_over_c2' in df.columns else df.get('gravitational_potential_ratio', np.zeros(len(df))))
    y = np.array(df['delta_alpha_over_alpha'])
    e = np.clip(np.array(df['error']), 1e-18, None)
    # Simple weighted linear fit y ≈ k * x
    w = 1.0 / e
    k = np.sum(w * x * y) / np.sum(w * x**2 + 1e-30)
    return k


def main():
    df = load_data()
    beta = fit_beta(df)
    print(f"Fitted β ≈ {beta:.3e}")
    plt.figure(figsize=(6,4))
    x = df['gravitational_potential_ratio'] if 'gravitational_potential_ratio' in df.columns else df['delta_phi_over_c2']
    plt.errorbar(x*1e10, df['delta_alpha_over_alpha']*1e18, yerr=df['error']*1e18, fmt='o', capsize=4)
    xline = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    yline = beta * xline
    plt.plot(xline*1e10, yline*1e18, 'r-', label='MRRC linear')
    plt.xlabel('Φ/c² (×10⁻¹⁰)')
    plt.ylabel('Δα/α (×10⁻¹⁸)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('atomic_clocks_fit.png', dpi=200)
    plt.close()
    print('Saved: atomic_clocks_fit.png')

if __name__ == '__main__':
    main()
