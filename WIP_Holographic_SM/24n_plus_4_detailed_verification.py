#!/usr/bin/env python3
"""
DETAILED VERIFICATION OF 24n+4 PATTERN
=======================================

Formula: M = 4 + 24n

For each particle, we:
1. Start with experimental mass M (from PDG 2024)
2. Solve for n: n = (M - 4) / 24
3. Round to nearest integer: n_best = round(n)
4. Calculate theoretical prediction: M_theory = 4 + 24*n_best
5. Calculate error: |M - M_theory| / M * 100%

Let's verify EVERY calculation manually.
"""

import numpy as np

def verify_particle(name, mass_mev, show_work=True):
    """
    Manually verify the 24n+4 formula for a single particle.
    
    Shows complete step-by-step calculation.
    """
    print(f"\n{'='*80}")
    print(f"PARTICLE: {name}")
    print(f"{'='*80}")
    print(f"Experimental mass: M = {mass_mev:.2f} MeV")
    print()
    
    # Step 1: Solve for n
    n_exact = (mass_mev - 4) / 24
    print(f"Step 1: Solve for n")
    print(f"  Formula: n = (M - 4) / 24")
    print(f"  n = ({mass_mev:.2f} - 4) / 24")
    print(f"  n = {mass_mev - 4:.2f} / 24")
    print(f"  n = {n_exact:.6f}")
    print()
    
    # Step 2: Round to nearest integer
    n_best = round(n_exact)
    print(f"Step 2: Round to nearest integer")
    print(f"  n_best = round({n_exact:.6f}) = {n_best}")
    print()
    
    # Step 3: Calculate theoretical mass
    mass_theory = 4 + 24 * n_best
    print(f"Step 3: Calculate theoretical mass")
    print(f"  M_theory = 4 + 24*n")
    print(f"  M_theory = 4 + 24*{n_best}")
    print(f"  M_theory = 4 + {24*n_best}")
    print(f"  M_theory = {mass_theory} MeV")
    print()
    
    # Step 4: Calculate error
    error_mev = mass_mev - mass_theory
    error_percent = abs(error_mev / mass_mev * 100)
    print(f"Step 4: Calculate error")
    print(f"  Error = M_exp - M_theory")
    print(f"  Error = {mass_mev:.2f} - {mass_theory}")
    print(f"  Error = {error_mev:+.2f} MeV")
    print(f"  Error% = |{error_mev:.2f}| / {mass_mev:.2f} × 100%")
    print(f"  Error% = {error_percent:.4f}%")
    print()
    
    # Step 5: Decision
    tolerance = 5.0  # 5% tolerance
    is_match = error_percent < tolerance
    print(f"Step 5: Within tolerance?")
    print(f"  Tolerance = ±{tolerance}%")
    print(f"  {error_percent:.4f}% < {tolerance}%? {is_match}")
    if is_match:
        print(f"  ✓ YES - This particle FITS the 24n+4 pattern!")
    else:
        print(f"  ✗ NO - This particle does NOT fit the pattern")
    
    return n_best, mass_theory, error_percent, is_match


def main():
    print("="*80)
    print("MANUAL VERIFICATION OF 24n+4 FORMULA")
    print("="*80)
    print()
    print("We will verify each particle claimed to fit the pattern.")
    print("All calculations shown in complete detail.")
    print()
    
    # Test cases - let's verify the most important ones
    test_particles = [
        # Perfect matches claimed
        ("Bottom quark", 4180.0),
        ("Top quark", 172760.0),
        ("W boson", 80379.0),
        ("Higgs boson", 125100.0),
        ("Z boson", 91188.0),
        ("Upsilon meson", 9460.30),
        
        # Near-perfect matches
        ("Proton", 938.27),
        ("Neutron", 939.57),
        ("Charm quark", 1275.0),
        
        # The original MRRC predictions
        ("MRRC n=3 prediction", 76.0),
        ("MRRC n=4 prediction", 100.0),
        ("MRRC n=5 prediction", 124.0),
        ("MRRC n=6 prediction", 148.0),
        ("MRRC n=7 prediction", 172.0),
        ("MRRC n=8 prediction", 196.0),
        
        # Particles that DON'T fit
        ("Electron", 0.511),
        ("Muon", 105.66),
        ("Up quark", 2.2),
        ("Down quark", 4.7),
    ]
    
    matches = []
    non_matches = []
    
    for name, mass in test_particles:
        n, theory, error, is_match = verify_particle(name, mass)
        
        if is_match:
            matches.append((name, mass, n, theory, error))
        else:
            non_matches.append((name, mass, n, theory, error))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF VERIFICATION")
    print("="*80)
    print()
    
    print(f"Particles tested: {len(test_particles)}")
    print(f"Matches (within 5%): {len(matches)}")
    print(f"Non-matches: {len(non_matches)}")
    print()
    
    print("MATCHES:")
    print(f"{'Particle':<25} {'M_exp':<12} {'n':<6} {'M_theory':<12} {'Error%':<10}")
    print("-"*80)
    for name, mass, n, theory, error in sorted(matches, key=lambda x: x[4]):
        print(f"{name:<25} {mass:>11.2f} {n:>6} {theory:>11.2f} {error:>9.4f}%")
    
    print()
    print("NON-MATCHES:")
    print(f"{'Particle':<25} {'M_exp':<12} {'n':<6} {'M_theory':<12} {'Error%':<10}")
    print("-"*80)
    for name, mass, n, theory, error in sorted(non_matches, key=lambda x: x[4]):
        print(f"{name:<25} {mass:>11.2f} {n:>6} {theory:>11.2f} {error:>9.4f}%")
    
    # Now verify the ORIGINAL claim about MRRC numbers
    print("\n" + "="*80)
    print("VERIFICATION OF ORIGINAL MRRC CLAIM")
    print("="*80)
    print()
    print("Original claim: 76, 100, 124, 148, 172, 196 are multiples of 24 plus 4")
    print()
    
    mrrc_values = [76, 100, 124, 148, 172, 196]
    for val in mrrc_values:
        n = (val - 4) / 24
        print(f"{val} = 4 + 24*{n:.1f}")
        if n == int(n):
            print(f"  ✓ EXACT: {val} = 4 + 24*{int(n)} = {4 + 24*int(n)}")
        else:
            print(f"  ✗ NOT EXACT: n = {n} is not an integer!")
    
    print("\n" + "="*80)
    print("CHECKING STANDARD MODEL PARTICLE MASSES")
    print("="*80)
    print()
    print("Do the MRRC predictions (76, 100, 124, 148, 172, 196 MeV) match")
    print("any known Standard Model particle masses?")
    print()
    
    # Known SM particles in MeV
    sm_particles = {
        'Electron': 0.511,
        'Muon': 105.66,
        'Tau': 1776.86,
        'Up quark': 2.2,
        'Down quark': 4.7,
        'Strange quark': 95.0,
        'Charm quark': 1275.0,
        'Bottom quark': 4180.0,
        'Top quark': 172760.0,
        'Pion±': 139.57,
        'Pion⁰': 134.98,
        'Kaon±': 493.68,
        'Proton': 938.27,
        'Neutron': 939.57,
    }
    
    print("MRRC predictions vs SM particle masses:")
    for mrrc_val in mrrc_values:
        print(f"\n{mrrc_val} MeV:")
        found_match = False
        for particle, mass in sm_particles.items():
            diff = abs(mass - mrrc_val)
            percent_diff = diff / mass * 100 if mass > 0 else 999
            if percent_diff < 10:  # Within 10%
                print(f"  {particle}: {mass:.2f} MeV (Δ = {diff:.2f} MeV, {percent_diff:.1f}%)")
                found_match = True
        if not found_match:
            print(f"  No SM particle within 10% of this value")


if __name__ == '__main__':
    main()
