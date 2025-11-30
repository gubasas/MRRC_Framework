"""
MRRC Markov Toy (didactic)
- Two-state Markov process with time-varying rates
- Demonstrates: maintenance vs chargeable change (mode-locked), weak-field β·Φ coupling, active drive, and expansion events

This is an illustrative model, not a physical MRRC simulator.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class MarkovConfig:
    steps: int = 3000
    dt: float = 0.01
    # Rates
    lambda0: float = 0.5           # baseline symmetric rate
    drive_amp: float = 0.6         # bias amplitude (fraction) during drive
    drive_period: float = 3.0      # seconds
    drive_duty: float = 0.5        # fraction active
    # Weak-field coupling (visual)
    beta: float = 7.27e-8
    phi_over_c2: float = 3.3e-10
    weak_visual_scale: float = 1e6  # visible factor; purely didactic
    # Mode-locked accounting
    mode_locked: bool = True
    # Expansion events (parameter jump)
    expansion_every: int | None = 600
    expansion_rate_boost: float = 0.2
    # Seed
    seed: int = 123


def drive_on(t: float, period: float, duty: float) -> bool:
    if period <= 0 or duty <= 0:
        return False
    phase = t % period
    return phase < duty * period


def simulate(cfg: MarkovConfig):
    rng = np.random.default_rng(cfg.seed)
    steps = cfg.steps
    dt = cfg.dt
    t = np.arange(steps) * dt
    p = np.zeros((steps, 2), dtype=float)
    p[0] = np.array([1.0, 0.0])  # start in state 0

    # Precompute weak-field factor
    f_beta = 1.0 + np.clip(cfg.beta * abs(cfg.phi_over_c2) * cfg.weak_visual_scale, 0.0, 0.5)

    legacy_cost = np.zeros(steps)
    change_cost = np.zeros(steps)
    S = np.zeros(steps)
    dSdt = np.zeros(steps)

    lam = cfg.lambda0

    def entropy(v: np.ndarray) -> float:
        v = np.clip(v, 1e-12, 1.0)
        return -np.sum(v * np.log(v))

    S[0] = entropy(p[0])

    for k in range(steps - 1):
        tk = t[k]
        active = drive_on(tk, cfg.drive_period, cfg.drive_duty)

        # Asymmetric bias during drive to force cycling; zero otherwise
        bias = cfg.drive_amp if active else 0.0

        # Weak-field visual factor scales activity
        lam_eff = lam * f_beta * (1.0 + (cfg.expansion_rate_boost if (cfg.expansion_every and k > 0 and k % cfg.expansion_every == 0) else 0.0))

        # Transition rates 0->1 and 1->0
        r01 = lam_eff * (1.0 + bias)
        r10 = lam_eff * (1.0 - bias)

        # Master equation (explicit Euler)
        p0, p1 = p[k]
        dp0 = -r01 * p0 + r10 * p1
        dp1 = -r10 * p1 + r01 * p0
        p[k+1, 0] = p0 + dt * dp0
        p[k+1, 1] = p1 + dt * dp1
        # Renormalize and clamp
        s = p[k+1].sum()
        if s <= 0:
            p[k+1] = p[k]
        else:
            p[k+1] = np.clip(p[k+1] / s, 1e-12, 1.0)

        S[k+1] = entropy(p[k+1])
        dSdt[k] = (S[k+1] - S[k]) / dt

        # Costs
        delta = np.abs(p[k+1] - p[k]).sum()
        legacy_cost[k+1] = legacy_cost[k] + delta
        charge = 0.0
        if cfg.mode_locked:
            exp_event = (cfg.expansion_every and k > 0 and k % cfg.expansion_every == 0)
            if active or exp_event:
                charge = delta
        else:
            charge = delta
        change_cost[k+1] = change_cost[k] + charge

    out = {
        't': t,
        'p': p,
        'S': S,
        'dSdt': dSdt,
        'legacy_cost': legacy_cost,
        'change_cost': change_cost,
        'f_beta': f_beta,
        'config': cfg,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description='MRRC Markov toy (didactic, not physical)')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--dt', type=float, default=0.01)
    ap.add_argument('--lambda0', type=float, default=0.5)
    ap.add_argument('--drive-amp', type=float, default=0.6)
    ap.add_argument('--drive-period', type=float, default=3.0)
    ap.add_argument('--drive-duty', type=float, default=0.5)
    ap.add_argument('--beta', type=float, default=7.27e-8)
    ap.add_argument('--phi', type=float, default=3.3e-10)
    ap.add_argument('--weak-scale', type=float, default=1e6)
    ap.add_argument('--mode-locked', action='store_true', default=True)
    ap.add_argument('--no-mode-locked', dest='mode_locked', action='store_false')
    ap.add_argument('--expand-every', type=int, default=600)
    ap.add_argument('--expand-boost', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()

    cfg = MarkovConfig(
        steps=args.steps,
        dt=args.dt,
        lambda0=args.lambda0,
        drive_amp=args.drive_amp,
        drive_period=args.drive_period,
        drive_duty=args.drive_duty,
        beta=args.beta,
        phi_over_c2=args.phi,
        weak_visual_scale=args.weak_scale,
        mode_locked=args.mode_locked,
        expansion_every=args.expand_every if args.expand_every > 0 else None,
        expansion_rate_boost=args.expand_boost,
        seed=args.seed,
    )

    out = simulate(cfg)
    t = out['t']
    S = out['S']
    dSdt = out['dSdt']
    L = out['legacy_cost']
    C = out['change_cost']

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t, S, label='Entropy S(t)')
    ax[0].plot(t, dSdt, label='dS/dt', alpha=0.7)
    ax[0].set_ylabel('Entropy / Rate (arb)')
    ax[0].legend()
    ax[0].set_title('MRRC Markov Toy (illustrative)')

    ax[1].plot(t, L, label='Legacy maintenance (cumulative)')
    ax[1].plot(t, C, label='Chargeable change (cumulative)')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Cost (arb)')
    ax[1].legend()
    fig.tight_layout()
    plt.savefig('mrrc_markov_sim.png', dpi=200)
    plt.close()

    print('Saved: mrrc_markov_sim.png')
    print(f"Weak-field factor f_beta (visual): {out['f_beta']:.3e}")
    print('Note: Didactic model; parameters exaggerated for visibility.')


if __name__ == '__main__':
    main()
