"""
Fe Kα line latency analysis (placeholder)
- Sketch: Load X-ray spectra, fit line profiles, compare residuals with MRRC busy-substrate index
"""
import numpy as np
import matplotlib.pyplot as plt

def mrrc_refractive_index(mu):
    return 1.0 + mu

def demo():
    # Placeholder synthetic line
    E = np.linspace(6.0, 7.5, 400)
    true_line = np.exp(-0.5 * ((E-6.4)/0.2)**2)
    mu = 0.02
    n = mrrc_refractive_index(mu)
    shifted = np.exp(-0.5 * ((E-6.4*n)/0.2)**2)
    plt.figure(figsize=(6,4))
    plt.plot(E, true_line, label='Geometric-only')
    plt.plot(E, shifted, label='MRRC busy-substrate')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Intensity (arb)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fe_ka_latency_demo.png', dpi=200)
    plt.close()
    print('Saved: fe_ka_latency_demo.png')

if __name__ == '__main__':
    demo()
