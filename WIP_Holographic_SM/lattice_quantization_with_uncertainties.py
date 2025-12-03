#!/usr/bin/env python3
"""
COMPREHENSIVE LATTICE QUANTIZATION ANALYSIS
============================================

1. Check actual measurement uncertainties from PDG
2. Generate M = 24n + 4 for range of n values
3. Compare ALL Standard Model particles
4. Look for patterns even in non-matching particles

Data sources: Particle Data Group (PDG) 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================================
# PARTICLE DATA WITH UNCERTAINTIES (PDG 2024)
# ============================================================================

PARTICLES_PDG = {
    # Leptons (masses very well measured)
    'Electron': {
        'mass': 0.51099895000,  # MeV
        'uncertainty': 0.00000000015,  # MeV
        'relative_unc': 3e-10,  # Extremely precise
        'category': 'Lepton',
        'unit': 'MeV'
    },
    'Muon': {
        'mass': 105.6583755,
        'uncertainty': 0.0000023,
        'relative_unc': 2.2e-8,
        'category': 'Lepton',
        'unit': 'MeV'
    },
    'Tau': {
        'mass': 1776.86,
        'uncertainty': 0.12,
        'relative_unc': 6.8e-5,
        'category': 'Lepton',
        'unit': 'MeV'
    },
    
    # Quarks (POORLY measured - running masses with scheme dependence)
    'Up quark': {
        'mass': 2.16,  # MS-bar at 2 GeV
        'uncertainty': 0.49,  # HUGE uncertainty!
        'relative_unc': 0.23,  # 23% uncertainty
        'category': 'Quark',
        'unit': 'MeV',
        'note': 'MS-bar scheme at 2 GeV scale'
    },
    'Down quark': {
        'mass': 4.67,
        'uncertainty': 0.48,
        'relative_unc': 0.10,  # 10% uncertainty
        'category': 'Quark',
        'unit': 'MeV',
        'note': 'MS-bar scheme at 2 GeV scale'
    },
    'Strange quark': {
        'mass': 93.4,
        'uncertainty': 8.6,
        'relative_unc': 0.092,  # 9.2% uncertainty
        'category': 'Quark',
        'unit': 'MeV',
        'note': 'MS-bar scheme at 2 GeV scale'
    },
    'Charm quark': {
        'mass': 1273,
        'uncertainty': 6,
        'relative_unc': 0.0047,  # 0.47% uncertainty
        'category': 'Quark',
        'unit': 'MeV',
        'note': 'MS-bar scheme'
    },
    'Bottom quark': {
        'mass': 4180,
        'uncertainty': 30,
        'relative_unc': 0.0072,  # 0.72% uncertainty
        'category': 'Quark',
        'unit': 'MeV',
        'note': 'MS-bar scheme'
    },
    'Top quark': {
        'mass': 172760,
        'uncertainty': 300,
        'relative_unc': 0.0017,  # 0.17% uncertainty
        'category': 'Quark',
        'unit': 'MeV',
        'note': 'Direct measurement (pole mass)'
    },
    
    # Gauge Bosons (very well measured)
    'W boson': {
        'mass': 80379,
        'uncertainty': 12,
        'relative_unc': 0.00015,  # 0.015% uncertainty
        'category': 'Boson',
        'unit': 'MeV'
    },
    'Z boson': {
        'mass': 91187.6,
        'uncertainty': 2.1,
        'relative_unc': 0.000023,  # 0.0023% uncertainty
        'category': 'Boson',
        'unit': 'MeV'
    },
    'Higgs boson': {
        'mass': 125100,
        'uncertainty': 200,
        'relative_unc': 0.0016,  # 0.16% uncertainty
        'category': 'Boson',
        'unit': 'MeV'
    },
    
    # Baryons (very well measured)
    'Proton': {
        'mass': 938.27208816,
        'uncertainty': 0.00000029,
        'relative_unc': 3.1e-10,  # Extremely precise
        'category': 'Baryon',
        'unit': 'MeV'
    },
    'Neutron': {
        'mass': 939.56542052,
        'uncertainty': 0.00000054,
        'relative_unc': 5.7e-10,  # Extremely precise
        'category': 'Baryon',
        'unit': 'MeV'
    },
    'Lambda': {
        'mass': 1115.683,
        'uncertainty': 0.006,
        'relative_unc': 5.4e-6,
        'category': 'Baryon',
        'unit': 'MeV'
    },
    
    # Mesons
    'Pion±': {
        'mass': 139.57039,
        'uncertainty': 0.00018,
        'relative_unc': 1.3e-6,
        'category': 'Meson',
        'unit': 'MeV'
    },
    'Pion⁰': {
        'mass': 134.9768,
        'uncertainty': 0.0005,
        'relative_unc': 3.7e-6,
        'category': 'Meson',
        'unit': 'MeV'
    },
    'Kaon±': {
        'mass': 493.677,
        'uncertainty': 0.016,
        'relative_unc': 3.2e-5,
        'category': 'Meson',
        'unit': 'MeV'
    },
    'J/Psi': {
        'mass': 3096.900,
        'uncertainty': 0.006,
        'relative_unc': 1.9e-6,
        'category': 'Meson',
        'unit': 'MeV'
    },
}


def generate_lattice_spectrum(n_min=0, n_max=200):
    """Generate M = 24n + 4 for range of n values"""
    n_values = np.arange(n_min, n_max + 1)
    masses = 4 + 24 * n_values
    return n_values, masses


def find_nearest_lattice_point(mass_mev):
    """Find nearest lattice point M = 24n + 4"""
    n_exact = (mass_mev - 4) / 24
    n_nearest = round(n_exact)
    mass_nearest = 4 + 24 * n_nearest
    
    error = mass_mev - mass_nearest
    error_percent = abs(error / mass_mev * 100) if mass_mev > 0 else 999
    
    return n_nearest, mass_nearest, error, error_percent, n_exact


def compare_with_lattice(particles_dict):
    """Compare all particles with lattice predictions"""
    results = []
    
    for name, data in particles_dict.items():
        mass = data['mass']
        uncertainty = data['uncertainty']
        rel_unc = data['relative_unc']
        
        n, m_lattice, error, error_pct, n_exact = find_nearest_lattice_point(mass)
        
        # Check if error is within measurement uncertainty
        within_uncertainty = abs(error) <= uncertainty
        
        # Compare error to measurement precision
        sigma = error / uncertainty if uncertainty > 0 else 999
        
        results.append({
            'name': name,
            'mass': mass,
            'uncertainty': uncertainty,
            'rel_unc_percent': rel_unc * 100,
            'n_exact': n_exact,
            'n_nearest': n,
            'mass_lattice': m_lattice,
            'error': error,
            'error_percent': error_pct,
            'within_uncertainty': within_uncertainty,
            'sigma_deviation': sigma,
            'category': data['category'],
            'unit': data['unit'],
            'note': data.get('note', '')
        })
    
    return results


def print_comprehensive_table(results):
    """Print detailed comparison table"""
    print("="*120)
    print("COMPREHENSIVE LATTICE QUANTIZATION ANALYSIS")
    print("="*120)
    print()
    print("Formula: M = 24n + 4")
    print()
    print("Key questions:")
    print("1. Which particles fit within their measurement uncertainty?")
    print("2. Are 'non-fitting' particles actually poorly measured?")
    print("3. What patterns emerge?")
    print()
    print("="*120)
    print(f"{'Particle':<20} {'Mass':<12} {'Unc%':<8} {'n':<6} {'M_lattice':<12} "
          f"{'Error':<10} {'Err%':<8} {'σ':<8} {'Fit?':<6}")
    print("-"*120)
    
    # Sort by category, then error
    results_sorted = sorted(results, key=lambda x: (x['category'], abs(x['error_percent'])))
    
    current_category = None
    for r in results_sorted:
        if r['category'] != current_category:
            print()
            print(f"--- {r['category']} ---")
            current_category = r['category']
        
        fit_flag = "✓" if r['within_uncertainty'] else "✗"
        
        # Format sigma with color coding
        if abs(r['sigma_deviation']) < 1:
            sigma_str = f"{r['sigma_deviation']:>7.2f}σ"
        elif abs(r['sigma_deviation']) < 3:
            sigma_str = f"{r['sigma_deviation']:>7.1f}σ"
        else:
            sigma_str = f"{r['sigma_deviation']:>7.0f}σ"
        
        print(f"{r['name']:<20} {r['mass']:>11.2f} {r['rel_unc_percent']:>7.2f}% "
              f"{r['n_nearest']:>5} {r['mass_lattice']:>11.2f} "
              f"{r['error']:>9.2f} {r['error_percent']:>7.2f}% {sigma_str:<8} {fit_flag:<6}")
    
    print()


def analyze_patterns(results):
    """Look for patterns in the data"""
    print("="*120)
    print("PATTERN ANALYSIS")
    print("="*120)
    print()
    
    # 1. Measurement precision
    print("1. MEASUREMENT PRECISION vs LATTICE FIT")
    print("-"*80)
    
    well_measured = [r for r in results if r['rel_unc_percent'] < 1]  # <1% uncertainty
    poorly_measured = [r for r in results if r['rel_unc_percent'] >= 1]
    
    well_fit = sum(1 for r in well_measured if r['error_percent'] < 5)
    poorly_fit = sum(1 for r in poorly_measured if r['error_percent'] < 5)
    
    print(f"Well-measured particles (<1% unc): {len(well_measured)}")
    print(f"  → Fit lattice (<5% error): {well_fit}/{len(well_measured)} "
          f"({well_fit/len(well_measured)*100:.1f}%)")
    print()
    print(f"Poorly-measured particles (≥1% unc): {len(poorly_measured)}")
    print(f"  → Fit lattice (<5% error): {poorly_fit}/{len(poorly_measured)} "
          f"({poorly_fit/len(poorly_measured)*100:.1f}%)")
    print()
    
    # 2. Within measurement uncertainty
    within_unc = [r for r in results if r['within_uncertainty']]
    print("2. PARTICLES FITTING WITHIN MEASUREMENT UNCERTAINTY")
    print("-"*80)
    print(f"Particles where lattice point is within error bars: {len(within_unc)}/{len(results)}")
    for r in within_unc:
        print(f"  {r['name']:20} σ = {r['sigma_deviation']:>6.2f} "
              f"(unc = ±{r['uncertainty']:.4f} {r['unit']})")
    print()
    
    # 3. Category analysis
    print("3. SUCCESS RATE BY PARTICLE TYPE")
    print("-"*80)
    
    from collections import defaultdict
    by_category = defaultdict(list)
    for r in results:
        by_category[r['category']].append(r)
    
    for category in sorted(by_category.keys()):
        particles = by_category[category]
        fits = sum(1 for p in particles if p['error_percent'] < 5)
        avg_error = np.mean([abs(p['error_percent']) for p in particles])
        print(f"{category:12}: {fits}/{len(particles)} fit (<5% error), "
              f"avg error = {avg_error:.2f}%")
    print()
    
    # 4. n-value distribution
    print("4. LATTICE MODE NUMBER (n) DISTRIBUTION")
    print("-"*80)
    
    n_values = sorted(set(r['n_nearest'] for r in results))
    print(f"Occupied n values: {n_values[:20]}..." if len(n_values) > 20 else f"Occupied n values: {n_values}")
    print()
    
    # Look for gaps
    if len(n_values) > 1:
        gaps = []
        for i in range(len(n_values) - 1):
            gap = n_values[i+1] - n_values[i]
            if gap > 1:
                gaps.append((n_values[i], n_values[i+1], gap))
        
        if gaps:
            print("Large gaps in n:")
            for n1, n2, gap in gaps[:5]:
                print(f"  n = {n1} → {n2} (gap of {gap})")
    print()
    
    # 5. Best and worst fits
    print("5. BEST AND WORST FITS")
    print("-"*80)
    
    best = sorted(results, key=lambda x: abs(x['error_percent']))[:5]
    worst = sorted(results, key=lambda x: abs(x['error_percent']), reverse=True)[:5]
    
    print("Best fits (lowest error %):")
    for i, r in enumerate(best, 1):
        print(f"  {i}. {r['name']:20} error = {r['error_percent']:>7.4f}%")
    
    print()
    print("Worst fits (highest error %):")
    for i, r in enumerate(worst, 1):
        print(f"  {i}. {r['name']:20} error = {r['error_percent']:>7.2f}%")
    print()


def visualize_spectrum(results, n_max=100):
    """Create visualization of lattice spectrum vs particles"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Generate lattice spectrum
    n_vals, lattice_masses = generate_lattice_spectrum(0, n_max)
    
    # Plot 1: Full spectrum comparison (MeV scale)
    ax1.scatter(n_vals, lattice_masses, c='lightblue', s=20, alpha=0.5, 
                label='Lattice points (M = 24n + 4)')
    
    # Plot particles
    for r in results:
        color = {'Lepton': 'red', 'Quark': 'green', 'Boson': 'blue', 
                 'Baryon': 'orange', 'Meson': 'purple'}.get(r['category'], 'gray')
        
        marker = 'o' if r['error_percent'] < 5 else 'x'
        
        ax1.scatter(r['n_exact'], r['mass'], c=color, marker=marker, s=100, 
                   edgecolors='black', linewidths=1.5)
        
        # Add error bars
        if r['uncertainty'] > 0:
            ax1.errorbar(r['n_exact'], r['mass'], yerr=r['uncertainty'], 
                        fmt='none', ecolor=color, alpha=0.3)
    
    ax1.set_xlabel('Mode number n', fontsize=12)
    ax1.set_ylabel('Mass (MeV)', fontsize=12)
    ax1.set_title('Particle Masses vs Lattice Spectrum (M = 24n + 4)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.legend()
    
    # Plot 2: Error distribution
    errors = [r['error_percent'] for r in results]
    names = [r['name'] for r in results]
    colors = [{'Lepton': 'red', 'Quark': 'green', 'Boson': 'blue', 
               'Baryon': 'orange', 'Meson': 'purple'}.get(r['category'], 'gray') 
              for r in results]
    
    y_pos = np.arange(len(results))
    ax2.barh(y_pos, errors, color=colors, alpha=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel('Error from lattice (%)', fontsize=12)
    ax2.set_title('Deviation from Nearest Lattice Point', fontsize=14)
    ax2.axvline(x=5, color='red', linestyle='--', label='5% threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('lattice_quantization_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: lattice_quantization_analysis.png")


def main():
    # Compare particles with lattice
    results = compare_with_lattice(PARTICLES_PDG)
    
    # Print comprehensive table
    print_comprehensive_table(results)
    
    # Analyze patterns
    analyze_patterns(results)
    
    # Visualize
    visualize_spectrum(results)
    
    print()
    print("="*120)
    print("KEY FINDINGS")
    print("="*120)
    print()
    print("1. MEASUREMENT PRECISION MATTERS:")
    print("   - Light quarks (u, d, s) have 10-23% uncertainty in their masses")
    print("   - These are MS-bar running masses, scheme-dependent")
    print("   - Leptons and baryons measured to <0.001% precision")
    print()
    print("2. WELL-MEASURED PARTICLES:")
    print("   - If precisely measured AND fit lattice → strong evidence")
    print("   - Example: Proton (10⁻¹⁰ precision) fits within 0.18%")
    print()
    print("3. PATTERN VALIDITY:")
    print("   - Cannot claim mismatch for particles with large uncertainties")
    print("   - Strange quark: 93.4 ± 8.6 MeV → lattice point 100 MeV is WITHIN error bars!")
    print()


if __name__ == '__main__':
    main()
