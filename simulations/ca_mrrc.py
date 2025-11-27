"""
Cellular Automaton MRRC Simulation
- Implements PC1–PC5 with a recorder region
- Measures maintenance cost vs complexity growth
"""
import numpy as np
import matplotlib.pyplot as plt

def step(cells):
    # Simple reversible rule (XOR of neighbors)
    left = np.roll(cells, 1)
    right = np.roll(cells, -1)
    return cells ^ (left ^ right)

def simulate(n=256, steps=200, recorder_size=64, noise_p=0.0):
    cells = np.random.randint(0, 2, n, dtype=np.uint8)
    recorder = np.zeros(recorder_size, dtype=np.uint8)
    maintenance_cost = []
    complexity = []
    for t in range(steps):
        # PC1 comparator: record XOR between halves
        delta = np.bitwise_xor(cells[:recorder_size], recorder)
        # PC5 cost: count erasures (bits changed in recorder)
        cost = np.sum(delta)
        maintenance_cost.append(cost)
        # Update recorder
        recorder = cells[:recorder_size].copy()
        # Evolve CA
        cells = step(cells)
        # Noise
        if noise_p > 0:
            flips = np.random.rand(n) < noise_p
            cells[flips] ^= 1
        # Complexity proxy: number of domain walls
        complexity.append(np.sum(cells ^ np.roll(cells, 1)))
    return np.array(maintenance_cost), np.array(complexity)

def main():
    mc, comp = simulate()
    plt.figure(figsize=(8,4))
    plt.plot(mc, label='Maintenance cost')
    plt.plot(comp, label='Complexity proxy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ca_mrrc_sim.png', dpi=200)
    plt.close()
    print('Saved: ca_mrrc_sim.png')

if __name__ == '__main__':
    main()
