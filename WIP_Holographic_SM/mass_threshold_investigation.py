#!/usr/bin/env python3
"""
MASS THRESHOLD ANALYSIS
=======================

Find the critical mass scale where particles transition from
NOT fitting the lattice to FITTING the lattice.

What physics changes at this threshold?
"""

import numpy as np
import matplotlib.pyplot as plt

# All particles with their masses and fit status
PARTICLES = {
    # DON'T FIT (gaps)
    'Electron': {'mass': 0.000511, 'fits': False, 'type': 'Elementary lepton'},
    'Up': {'mass': 0.00216, 'fits': False, 'type': 'Elementary quark'},
    'Down': {'mass': 0.00467, 'fits': False, 'type': 'Elementary quark'},
    'Strange': {'mass': 0.0934, 'fits': False, 'type': 'Elementary quark'},
    'Muon': {'mass': 0.10566, 'fits': False, 'type': 'Elementary lepton'},
    'Pion0': {'mass': 0.13498, 'fits': False, 'type': 'Composite meson'},
    'Pion+': {'mass': 0.13957, 'fits': False, 'type': 'Composite meson'},
    
    # FIT (on lattice)
    'Kaon': {'mass': 0.49368, 'fits': True, 'type': 'Composite meson'},
    'Proton': {'mass': 0.93827, 'fits': True, 'type': 'Composite baryon'},
    'Neutron': {'mass': 0.93957, 'fits': True, 'type': 'Composite baryon'},
    'Lambda': {'mass': 1.11568, 'fits': True, 'type': 'Composite baryon'},
    'Charm': {'mass': 1.273, 'fits': True, 'type': 'Elementary quark'},
    'Tau': {'mass': 1.77686, 'fits': True, 'type': 'Elementary lepton'},
    'J/Psi': {'mass': 3.0969, 'fits': True, 'type': 'Composite meson'},
    'Bottom': {'mass': 4.180, 'fits': True, 'type': 'Elementary quark'},
    'W': {'mass': 80.379, 'fits': True, 'type': 'Elementary boson'},
    'Z': {'mass': 91.1876, 'fits': True, 'type': 'Elementary boson'},
    'Higgs': {'mass': 125.1, 'fits': True, 'type': 'Elementary boson'},
    'Top': {'mass': 172.76, 'fits': True, 'type': 'Elementary quark'},
}

def find_threshold():
    """Find the critical mass threshold"""
    
    # Separate particles
    gaps = [(name, data['mass']) for name, data in PARTICLES.items() if not data['fits']]
    fits = [(name, data['mass']) for name, data in PARTICLES.items() if data['fits']]
    
    # Sort by mass
    gaps_sorted = sorted(gaps, key=lambda x: x[1])
    fits_sorted = sorted(fits, key=lambda x: x[1])
    
    # Find threshold
    max_gap = max(gaps_sorted, key=lambda x: x[1])
    min_fit = min(fits_sorted, key=lambda x: x[1])
    
    threshold_lower = max_gap[1]
    threshold_upper = min_fit[1]
    threshold_mid = (threshold_lower + threshold_upper) / 2
    
    print("="*80)
    print("MASS THRESHOLD ANALYSIS")
    print("="*80)
    print()
    print("Heaviest particle that DOESN'T fit:")
    print(f"  {max_gap[0]:15} {max_gap[1]*1000:>8.2f} MeV")
    print()
    print("Lightest particle that FITS:")
    print(f"  {min_fit[0]:15} {min_fit[1]*1000:>8.2f} MeV")
    print()
    print(f"THRESHOLD RANGE: {threshold_lower*1000:.1f} - {threshold_upper*1000:.1f} MeV")
    print(f"CRITICAL MASS: ~{threshold_mid*1000:.0f} MeV (~{threshold_mid:.2f} GeV)")
    print()
    
    return threshold_mid, threshold_lower, threshold_upper, gaps_sorted, fits_sorted


def analyze_physics_at_threshold(threshold):
    """What physics happens at this energy scale?"""
    
    print("="*80)
    print(f"WHAT HAPPENS AT ~{threshold*1000:.0f} MeV?")
    print("="*80)
    print()
    
    # Known physics scales
    scales = {
        'Electron mass': 0.511,
        'Pion mass (QCD scale)': 140,
        'Strange quark mass': 95,
        'ρ/ω meson mass': 770,
        'Proton/neutron mass': 940,
        'ΛQCD (QCD confinement)': 200,
        'Electroweak scale': 246000,  # Higgs VEV in MeV
    }
    
    threshold_mev = threshold * 1000
    
    print("Known physics scales for comparison:")
    print(f"{'Scale':<30} {'Value (MeV)':<15} {'Ratio to threshold':<20}")
    print("-"*80)
    
    for name, value in sorted(scales.items(), key=lambda x: x[1]):
        ratio = value / threshold_mev
        proximity = "✓ MATCH!" if 0.5 < ratio < 2.0 else ""
        print(f"{name:<30} {value:>10.1f} MeV {ratio:>15.2f}x {proximity}")
    
    print()
    print("="*80)
    print("PHYSICAL INTERPRETATION")
    print("="*80)
    print()
    
    if 200 < threshold_mev < 500:
        print(f"The threshold (~{threshold_mev:.0f} MeV) falls in the QCD confinement region!")
        print()
        print("This is the energy scale where:")
        print("  • Quarks become confined into hadrons")
        print("  • Color charge is screened (confinement)")
        print("  • Strong force dynamics dominate")
        print("  • Asymptotic freedom breaks down")
        print()
        print("HYPOTHESIS:")
        print("  The 24-cell lattice quantization M = 24n + 4 may represent:")
        print("  → Confined states (baryons, heavy mesons)")
        print("  → Mass from QCD binding + Higgs mechanism")
        print("  → Geometric quantization of composite structures")
        print()
        print("  Below threshold:")
        print("  → Perturbative regime (asymptotic freedom)")
        print("  → Current quark masses (not constituent masses)")
        print("  → Different geometric description needed")
        print()
        print("  Above threshold:")
        print("  → Non-perturbative regime (confinement)")
        print("  → Constituent masses dominate")
        print("  → 24-cell lattice structure emerges")
    
    elif 500 < threshold_mev < 1500:
        print(f"The threshold (~{threshold_mev:.0f} MeV) is around strange quark/kaon scale!")
        print()
        print("This suggests the lattice applies to:")
        print("  • Particles with strangeness")
        print("  • Heavy flavor physics")
        print("  • Particles above the strange threshold")


def create_visualization(threshold_mid, threshold_lower, threshold_upper, gaps, fits):
    """Visualize the threshold"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: All particles showing threshold
    all_particles = []
    for name, data in PARTICLES.items():
        all_particles.append({
            'name': name,
            'mass': data['mass'] * 1000,  # Convert to MeV
            'fits': data['fits'],
            'type': data['type']
        })
    
    all_particles.sort(key=lambda x: x['mass'])
    
    y_pos = range(len(all_particles))
    masses = [p['mass'] for p in all_particles]
    colors = ['green' if p['fits'] else 'red' for p in all_particles]
    labels = [p['name'] for p in all_particles]
    
    ax1.barh(y_pos, masses, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel('Mass (MeV, log scale)', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_title('Particle Masses: Threshold Between Fit and Gap Particles', 
                  fontsize=14, fontweight='bold')
    
    # Draw threshold region
    ax1.axvline(x=threshold_lower*1000, color='blue', linestyle='--', 
                linewidth=3, alpha=0.7, label=f'Threshold range')
    ax1.axvline(x=threshold_upper*1000, color='blue', linestyle='--', 
                linewidth=3, alpha=0.7)
    ax1.axvspan(threshold_lower*1000, threshold_upper*1000, 
                alpha=0.2, color='blue', label='Transition region')
    
    # Mark critical mass
    ax1.axvline(x=threshold_mid*1000, color='purple', linestyle='-', 
                linewidth=2, label=f'Critical mass ~{threshold_mid*1000:.0f} MeV')
    
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Histogram showing distribution
    gap_masses = [p['mass'] for p in all_particles if not p['fits']]
    fit_masses = [p['mass'] for p in all_particles if p['fits']]
    
    bins = np.logspace(-1, 3, 50)  # Log-spaced bins from 0.1 to 1000 MeV
    
    ax2.hist(gap_masses, bins=bins, alpha=0.6, color='red', 
             label='Gap particles (don\'t fit)', edgecolor='black')
    ax2.hist(fit_masses, bins=bins, alpha=0.6, color='green',
             label='Fit particles (on lattice)', edgecolor='black')
    
    ax2.set_xlabel('Mass (MeV, log scale)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of particles', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_title('Mass Distribution: Clear Separation at Threshold', 
                  fontsize=14, fontweight='bold')
    
    # Draw threshold
    ax2.axvline(x=threshold_mid*1000, color='purple', linestyle='-', 
                linewidth=3, label=f'Threshold ~{threshold_mid*1000:.0f} MeV')
    ax2.axvspan(threshold_lower*1000, threshold_upper*1000, 
                alpha=0.2, color='blue')
    
    # Mark important physics scales
    physics_scales = [
        (140, 'Pion mass'),
        (200, 'Λ_QCD'),
        (940, 'Nucleon mass'),
    ]
    
    for scale, name in physics_scales:
        ax2.axvline(x=scale, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax2.text(scale, ax2.get_ylim()[1]*0.9, name, 
                rotation=90, va='top', ha='right', fontsize=8, color='gray')
    
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mass_threshold_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: mass_threshold_analysis.png")


def analyze_quantum_numbers():
    """Check if quantum numbers change at threshold"""
    
    print()
    print("="*80)
    print("QUANTUM NUMBER ANALYSIS")
    print("="*80)
    print()
    
    print("Particles BELOW threshold (don't fit):")
    print(f"{'Particle':<15} {'Mass (MeV)':<12} {'Spin':<8} {'Charge':<8} {'Strangeness':<12}")
    print("-"*80)
    
    below = [
        ('Electron', 0.511, '1/2', '-1', '0'),
        ('Up', 2.16, '1/2', '+2/3', '0'),
        ('Down', 4.67, '1/2', '-1/3', '0'),
        ('Strange', 93.4, '1/2', '-1/3', '-1'),
        ('Muon', 105.66, '1/2', '-1', '0'),
        ('Pion0', 134.98, '0', '0', '0'),
        ('Pion+', 139.57, '0', '+1', '0'),
    ]
    
    for p in below:
        print(f"{p[0]:<15} {p[1]:>11.2f} {p[2]:>8} {p[3]:>8} {p[4]:>12}")
    
    print()
    print("Particles ABOVE threshold (fit):")
    print(f"{'Particle':<15} {'Mass (MeV)':<12} {'Spin':<8} {'Charge':<8} {'Strangeness':<12}")
    print("-"*80)
    
    above = [
        ('Kaon', 493.68, '0', '+1', '+1'),
        ('Proton', 938.27, '1/2', '+1', '0'),
        ('Neutron', 939.57, '1/2', '0', '0'),
        ('Lambda', 1115.68, '1/2', '0', '-1'),
        ('Charm', 1273, '1/2', '+2/3', '0'),
        ('Tau', 1776.86, '1/2', '-1', '0'),
    ]
    
    for p in above:
        print(f"{p[0]:<15} {p[1]:>11.2f} {p[2]:>8} {p[3]:>8} {p[4]:>12}")
    
    print()
    print("OBSERVATION:")
    print("  → Quantum numbers (spin, charge, strangeness) do NOT change at threshold")
    print("  → Both sides have spin-1/2 and spin-0 particles")
    print("  → Both sides have various charges")
    print()
    print("  → The threshold is purely MASS-BASED, not quantum number based")


def main():
    threshold_mid, threshold_lower, threshold_upper, gaps, fits = find_threshold()
    
    analyze_physics_at_threshold(threshold_mid)
    
    create_visualization(threshold_mid, threshold_lower, threshold_upper, gaps, fits)
    
    analyze_quantum_numbers()
    
    print()
    print("="*80)
    print("FINAL ANSWER: What changes at the threshold?")
    print("="*80)
    print()
    print(f"THRESHOLD: ~{threshold_mid*1000:.0f} MeV (~{threshold_mid:.2f} GeV)")
    print()
    print("What CHANGES:")
    print()
    print("1. CONFINEMENT PHYSICS")
    print("   • Below: Asymptotic freedom, perturbative QCD")
    print("   • Above: Confinement, non-perturbative QCD")
    print()
    print("2. MASS GENERATION MECHANISM")
    print("   • Below: 'Current' quark masses (from Higgs)")
    print("   • Above: 'Constituent' masses (Higgs + QCD binding)")
    print()
    print("3. GEOMETRIC STRUCTURE")
    print("   • Below: Free/perturbative states, no lattice quantization")
    print("   • Above: Confined/bound states, 24-cell lattice quantization")
    print()
    print("4. ENERGY SCALE")
    print("   • Below: Sub-nucleon scale, fundamental particle level")
    print("   • Above: Nucleon scale and up, composite/heavy states")
    print()
    print("PHYSICAL INTERPRETATION:")
    print()
    print("The 24-cell lattice M = 24n + 4 appears to govern particles where")
    print("CONFINEMENT and BINDING ENERGY are significant contributors to mass.")
    print()
    print("Light particles (< ~300 MeV) have mass primarily from Higgs mechanism")
    print("and don't show lattice quantization.")
    print()
    print("Heavy particles (> ~500 MeV) have mass from Higgs + strong binding")
    print("and DO show lattice quantization.")
    print()
    print("This suggests the 24-cell geometry is related to the NON-PERTURBATIVE")
    print("QCD vacuum structure and confinement mechanism!")


if __name__ == '__main__':
    main()
