"""
Cellular Automaton MRRC Simulation (V5.1-aligned)
- Implements PC1–PC5 proxies with a recorder region
- Separates legacy maintenance cost from chargeable change cost (mode-locked substrate)
- Adds weak-field coupling via β and Φ/c², optional active drive, and expansion schedule
"""
import numpy as np
import matplotlib.pyplot as plt
import zlib
from dataclasses import dataclass


@dataclass
class MRRCConfig:
    n: int = 256
    steps: int = 400
    recorder_size: int = 64
    # Weak-field coupling (lab-scale defaults)
    beta: float = 7.27e-8
    phi_over_c2: float = 3.3e-10  # seasonal Earth–Sun amplitude (abs)
    # Stochastic noise baseline (dimensionless flip probability)
    noise_p: float = 0.0
    # Optional active drive (ISS/ACES-like periodic perturbation)
    drive_amp: float = 0.0         # additional flip prob when drive is on
    drive_period: int | None = None  # steps between drive bursts
    drive_duty: float = 0.5        # fraction of period active
    # Mode-locked substrate: maintenance free; pay for change (expansion/drive)
    mode_locked: bool = True
    # Expansion schedule (universe growth)
    expansion_every: int | None = None  # steps cadence; None disables
    expansion_size: int = 0             # cells added to recorder when expanding
    # Reproducibility
    seed: int | None = 42

def step(cells):
    # Simple reversible rule (XOR of neighbors)
    left = np.roll(cells, 1)
    right = np.roll(cells, -1)
    return cells ^ (left ^ right)

def _lz_complexity(bits: np.ndarray) -> float:
    # Very rough proxy via zlib compression ratio
    by = np.packbits(bits).tobytes()
    if not by:
        return 0.0
    comp = zlib.compress(by, level=9)
    return len(comp) / max(1, len(by))


def simulate(cfg: MRRCConfig):
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n
    cells = rng.integers(0, 2, n, dtype=np.uint8)
    rec_size = cfg.recorder_size
    recorder = np.zeros(rec_size, dtype=np.uint8)

    # Effective weak-field flip probability from β·|Φ| (scaled up for visibility)
    # Note: true lab-scale rates ~ 1e-17 are visually imperceptible; scale for demo
    weak_field_scale = 1e4  # purely for visualization
    p_beta = np.clip(cfg.beta * abs(cfg.phi_over_c2) * weak_field_scale, 0.0, 0.05)

    legacy_maintenance = []        # PC5 (bit erasures in recorder)
    chargeable_change = []         # V5.1: only charge expansion/active drive
    domain_walls = []              # PC3 proxy (spatial complexity)
    lz_scores = []                 # PC3 proxy (compressibility)
    latency = []                   # PC4 proxy (lag to recorder match)

    lag_counter = 0
    last_match = True

    for t in range(cfg.steps):
        # Determine active drive state
        drive_on = False
        if cfg.drive_period and cfg.drive_period > 0:
            phase = t % cfg.drive_period
            drive_on = (phase < int(cfg.drive_duty * cfg.drive_period))

        # Recorders: compare and update legacy maintenance cost (PC5)
        view = cells[:rec_size]
        delta = np.bitwise_xor(view, recorder)
        cost = int(np.sum(delta))
        legacy_maintenance.append(cost)

        # Mode-locked chargeable change: only when expansion or active drive
        charge = 0
        if cfg.mode_locked:
            if (cfg.expansion_every and cfg.expansion_every > 0 and t % cfg.expansion_every == 0 and t > 0):
                charge += cost  # charge changes at expansion events
            if drive_on and cfg.drive_amp > 0:
                charge += cost  # charge during driven windows
        else:
            charge = cost
        chargeable_change.append(charge)

        # Update latency proxy (PC4): count frames until recorder matches view
        if cost == 0:
            if not last_match:
                latency.append(lag_counter)
                lag_counter = 0
            last_match = True
        else:
            last_match = False
            lag_counter += 1

        # Update recorder (PC1 comparator/recorder)
        recorder = view.copy()

        # Evolve CA
        cells = step(cells)

        # Apply stochastic flips: baseline noise + weak-field + active drive
        p_total = cfg.noise_p + p_beta + (cfg.drive_amp if drive_on else 0.0)
        if p_total > 0:
            flips = rng.random(n) < np.clip(p_total, 0.0, 0.5)
            cells[flips] ^= 1

        # Optional expansion: grow recorder window (universe growth)
        if cfg.expansion_every and cfg.expansion_every > 0 and t % cfg.expansion_every == 0 and t > 0:
            new_size = min(n, rec_size + max(0, cfg.expansion_size))
            if new_size > rec_size:
                new_rec = np.zeros(new_size, dtype=np.uint8)
                new_rec[:rec_size] = recorder
                recorder = new_rec
                rec_size = new_size

        # Complexity proxies
        domain_walls.append(int(np.sum(cells ^ np.roll(cells, 1))))
        lz_scores.append(_lz_complexity(cells))

    return {
        'legacy_maintenance': np.array(legacy_maintenance),
        'chargeable_change': np.array(chargeable_change),
        'domain_walls': np.array(domain_walls),
        'lz_complexity': np.array(lz_scores),
        'latency_samples': np.array(latency) if latency else np.array([]),
        'config': cfg,
        'p_beta_demo': p_beta,
    }

def main():
    cfg = MRRCConfig(
        n=256,
        steps=600,
        recorder_size=64,
        beta=7.27e-8,
        phi_over_c2=3.3e-10,
        noise_p=0.0,
        drive_amp=5e-3,       # visual active drive
        drive_period=60,
        drive_duty=0.33,
        mode_locked=True,
        expansion_every=100,
        expansion_size=8,
        seed=42,
    )
    out = simulate(cfg)

    t = np.arange(cfg.steps)
    plt.figure(figsize=(10,6))
    ax1 = plt.subplot(2,1,1)
    ax1.plot(t, out['legacy_maintenance'], label='Legacy maintenance (PC5)')
    ax1.plot(t, out['chargeable_change'], label='Chargeable change (V5.1)')
    ax1.set_ylabel('Cost (bits)')
    ax1.legend()
    ax1.set_title('MRRC CA Recorder: Maintenance vs Change Costs')

    ax2 = plt.subplot(2,1,2)
    ax2.plot(t, out['domain_walls'], label='Domain walls (PC3)')
    ax2.plot(t, out['lz_complexity']*100, label='LZ ratio ×100 (PC3)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Complexity (arb)')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('ca_mrrc_sim.png', dpi=200)
    plt.close()

    lat = out['latency_samples']
    p_beta = out['p_beta_demo']
    print('Saved: ca_mrrc_sim.png')
    print(f"Latency samples (PC4): n={len(lat)}, median={np.median(lat) if len(lat)>0 else 'NA'}")
    print(f"Demo weak-field flip prob from β·|Φ| (scaled): p_beta={p_beta:.3e}")

if __name__ == '__main__':
    main()
