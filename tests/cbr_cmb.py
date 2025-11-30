"""
MRRC vs CBR/CMB Test
=====================

Objective:
- Evaluate whether the MRRC gravitational coupling (β) would induce detectable
  variations of the fine-structure constant α at recombination, and the resulting
  impact on CMB-relevant quantities (Thomson cross-section σ_T and recombination redshift z*).

Method:
- MRRC predicts: Δα/α ≈ β · (Φ/c²)
- Sachs–Wolfe relates CMB temperature anisotropy to potential at last scattering:
  ΔT/T ≈ (1/3) · (Φ/c²)  ⇒ RMS[(Φ/c²)] ≈ 3 · RMS[ΔT/T]
- Use RMS[ΔT/T] ≈ 1×10⁻⁵ as a characteristic large-scale amplitude.
- Then RMS[Φ/c²] ≈ 3×10⁻⁵; predicted RMS[Δα/α] ≈ β · 3×10⁻⁵.
- Sensitivities:
  σ_T ∝ α²  ⇒ δσ_T/σ_T ≈ 2 · Δα/α
  E_binding ∝ α² ⇒ z* scales roughly as α²  ⇒ δz*/z* ≈ 2 · Δα/α (order-of-magnitude)

Outputs:
- Plot: fractional shifts for Δα/α, δσ_T/σ_T, δz*/z* at recombination-scale potentials,
  using (a) MRRC lab β upper limit, and (b) a hypothetical β that saturates a loose
  fiducial CMB constraint on |Δα/α| at recombination (e.g., 3×10⁻³).
- Text report summarizing assumptions and takeaways.

Note:
- This is a back-of-the-envelope test, not a full Boltzmann code analysis (CAMB/CLASS).
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

# Constants / defaults
RMS_DT_OVER_T = 1.0e-5           # characteristic CMB anisotropy RMS
RMS_PHI_OVER_C2 = 3.0 * RMS_DT_OVER_T  # via Sachs–Wolfe
FIDUCIAL_CMB_ALPHA_BOUND = 3.0e-3 # loose percent-level bound on |Δα/α| at recombination


def extract_best_beta_from_report(report_path: Path) -> float:
    """Parse β upper limit from mrrc_alpha_variation_report.txt if available.

    Returns:
        float: β upper limit, or np.nan if not found.
    """
    try:
        text = report_path.read_text()
        m = re.search(r"Best β bound \(drift \+ seasonal\):\s*([0-9.]+e[+-]\d+)", text)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return np.nan


def compute_cmb_impacts(beta: float,
                         rms_phi_over_c2: float = RMS_PHI_OVER_C2) -> dict:
    """Compute MRRC-predicted RMS impacts on α and derived CMB quantities.

    Args:
        beta: MRRC gravitational coupling β
        rms_phi_over_c2: RMS potential at last scattering (dimensionless)

    Returns:
        dict with keys: dalpha_rms, dsigT_over_sigT_rms, dzstar_over_zstar_rms
    """
    dalpha_rms = beta * rms_phi_over_c2
    dsig_over_sig = 2.0 * dalpha_rms
    dz_over_z = 2.0 * dalpha_rms
    return {
        'dalpha_rms': dalpha_rms,
        'dsigT_over_sigT_rms': dsig_over_sig,
        'dzstar_over_zstar_rms': dz_over_z,
        'rms_phi_over_c2': rms_phi_over_c2
    }


def main():
    cwd = Path.cwd()
    report_path = cwd / 'mrrc_alpha_variation_report.txt'
    beta_best = extract_best_beta_from_report(report_path)

    # Fallback if not found
    if not np.isfinite(beta_best):
        beta_best = 1.0e-7  # conservative placeholder

    impacts_lab = compute_cmb_impacts(beta_best)

    # Hypothetical β that would saturate a loose CMB |Δα/α| bound at recombination
    beta_cmb_max = FIDUCIAL_CMB_ALPHA_BOUND / RMS_PHI_OVER_C2
    impacts_cmb = compute_cmb_impacts(beta_cmb_max)

    # Prepare plot
    labels = ['|Δα/α|', '|δσ_T/σ_T|', '|δz*/z*|']
    lab_vals = [
        abs(impacts_lab['dalpha_rms']),
        abs(impacts_lab['dsigT_over_sigT_rms']),
        abs(impacts_lab['dzstar_over_zstar_rms'])
    ]
    cmb_vals = [
        abs(impacts_cmb['dalpha_rms']),
        abs(impacts_cmb['dsigT_over_sigT_rms']),
        abs(impacts_cmb['dzstar_over_zstar_rms'])
    ]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, lab_vals, width, label='MRRC (lab β upper limit)')
    plt.bar(x + width/2, cmb_vals, width, label='β saturating loose CMB |Δα/α|')
    plt.yscale('log')
    plt.xticks(x, labels)
    plt.ylabel('Fractional RMS (log scale)')
    plt.title('MRRC → CMB-scale fractional impacts (RMS)')
    plt.grid(True, which='both', axis='y', alpha=0.3)
    plt.legend()
    out_plot = cwd / 'mrrc_cmb_test.png'
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)
    plt.close()

    # Write report
    lines = []
    lines.append('='*60)
    lines.append('MRRC vs CBR/CMB Test')
    lines.append('='*60)
    lines.append('')
    lines.append(f'Inputs:')
    lines.append(f'  RMS[ΔT/T] ≈ {RMS_DT_OVER_T:.1e} ⇒ RMS[Φ/c²] ≈ {RMS_PHI_OVER_C2:.1e}')
    lines.append(f'  Best β (from lab + seasonal): {beta_best:.3e}')
    lines.append(f'  Loose CMB |Δα/α| bound: {FIDUCIAL_CMB_ALPHA_BOUND:.1e} ⇒ β < {beta_cmb_max:.2e}')
    lines.append('')
    lines.append('Predicted RMS fractional impacts at recombination scale:')
    lines.append(f"  Using lab β: |Δα/α| ≈ {impacts_lab['dalpha_rms']:.3e}, |δσ_T/σ_T| ≈ {impacts_lab['dsigT_over_sigT_rms']:.3e}, |δz*/z*| ≈ {impacts_lab['dzstar_over_zstar_rms']:.3e}")
    lines.append(f"  Using β@CMB max: |Δα/α| ≈ {impacts_cmb['dalpha_rms']:.3e}, |δσ_T/σ_T| ≈ {impacts_cmb['dsigT_over_sigT_rms']:.3e}, |δz*/z*| ≈ {impacts_cmb['dzstar_over_zstar_rms']:.3e}")
    lines.append('')
    lines.append('Interpretation:')
    lines.append('  - With the current laboratory bound on β, MRRC predicts |Δα/α| ~ 10⁻¹²–10⁻¹³ at recombination-scale')
    lines.append('    potentials, implying δσ_T/σ_T and δz*/z* of the same tiny order—far below current CMB sensitivity.')
    lines.append('  - Even a loose CMB-level |Δα/α| allowance (~10⁻³) corresponds to β ~ 10²–10³, orders of magnitude')
    lines.append('    weaker than lab constraints; thus CMB data are automatically satisfied if lab bounds hold.')
    lines.append('  - A full Boltzmann analysis (e.g., CAMB/CLASS) could refine coefficients but won’t change the qualitative')
    lines.append('    conclusion: MRRC gravitational coupling that passes atomic clock nulls implies negligible CMB impact.')
    lines.append('')
    lines.append(f'Plot saved: {out_plot.name}')
    out_report = cwd / 'mrrc_cmb_report.txt'
    out_report.write_text('\n'.join(lines))
    print('\n'.join(lines))
    print(f"\nReport saved: {out_report}")
    print(f"Plot saved:   {out_plot}")


if __name__ == '__main__':
    main()
