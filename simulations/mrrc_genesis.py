"""
MRRC-Genesis: energy-led emergence with locking and hierarchy
- Grid evolves from noise under stress/yield rule (V5.1-inspired)
- Global energy pays for active updates (dissipation tax)
- Locked regions accumulate hierarchy (stability age)
- Visualizes emergent structure from an initial energy budget

Usage:
  python3 simulations/mrrc_genesis.py --live
  python3 simulations/mrrc_genesis.py --outfile mrrc_genesis.gif --frames 300
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Defaults inspired by the user's proposal
DEFAULT_GRID_SIZE = 100
DEFAULT_INITIAL_ENERGY = 10000.0
DEFAULT_DISSIPATION_RATE = 0.01
DEFAULT_YIELD_POINT = 0.2
DEFAULT_COUPLING = 0.1

class MRRCUniverse:
    def __init__(self, size: int,
                 initial_energy: float,
                 dissipation_rate: float,
                 yield_point: float,
                 coupling: float,
                 seed: int | None = None):
        self.rng = np.random.default_rng(seed)  # keep RNG for noise injection
        self.size = size
        self.state = self.rng.random((size, size))
        self.hierarchy_level = np.zeros((size, size), dtype=np.float32)
        self.global_energy = float(initial_energy)
        self.dissipation_rate = float(dissipation_rate)
        self.yield_point = float(yield_point)
        self.coupling = float(coupling)
        self.time_step = 0
        self.noise_rate = 0.001  # small noise injection to prevent total freeze

    def evolve(self):
        if self.global_energy <= 0:
            return  # heat death

        # Inject tiny noise to prevent total freeze (quantum fluctuations analogue)
        if self.noise_rate > 0:
            noise = self.rng.normal(0, self.noise_rate, size=self.state.shape)
            self.state += noise
            np.clip(self.state, 0.0, 1.0, out=self.state)

        # Relational stress via simple gradients (4-neighborhood)
        grad_x = np.roll(self.state, 1, axis=0) - self.state
        grad_y = np.roll(self.state, 1, axis=1) - self.state
        stress = np.abs(grad_x) + np.abs(grad_y)

        # Stiffness filter: only cells above yield point can change
        active_mask = stress > self.yield_point
        num_changes = int(np.sum(active_mask))
        cost = num_changes * self.dissipation_rate

        if self.global_energy > cost and num_changes > 0:
            # Diffusive update (discrete Laplacian)
            laplacian = (
                np.roll(self.state, 1, axis=0) +
                np.roll(self.state, -1, axis=0) +
                np.roll(self.state, 1, axis=1) +
                np.roll(self.state, -1, axis=1) -
                4 * self.state
            )
            change = self.coupling * laplacian
            self.state[active_mask] += change[active_mask]
            # keep state in [0,1] for stable colormap
            np.clip(self.state, 0.0, 1.0, out=self.state)
            self.global_energy -= cost

        # Hierarchy accumulation for locked cells; reset for active ones
        self.hierarchy_level[~active_mask] += 1.0
        self.hierarchy_level[active_mask] = 0.0

        self.time_step += 1


def run_sim(size=DEFAULT_GRID_SIZE,
            initial_energy=DEFAULT_INITIAL_ENERGY,
            dissipation_rate=DEFAULT_DISSIPATION_RATE,
            yield_point=DEFAULT_YIELD_POINT,
            coupling=DEFAULT_COUPLING,
            frames=200, interval=50, seed=None,
            live=False, outfile='mrrc_genesis.gif'):
    uni = MRRCUniverse(size=size,
                       initial_energy=initial_energy,
                       dissipation_rate=dissipation_rate,
                       yield_point=yield_point,
                       coupling=coupling,
                       seed=seed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Persistent colorbars and legends
    from matplotlib.colors import Normalize
    norm_state = Normalize(vmin=0, vmax=1)
    norm_hier = Normalize(vmin=0, vmax=50)
    # Initial images for attaching colorbars
    im1 = ax1.imshow(uni.state, cmap='inferno', norm=norm_state)
    im2 = ax2.imshow(uni.hierarchy_level, cmap='magma', norm=norm_hier)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Information/Energy density (0 → 1)')
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Hierarchy age (ticks)')
    # Legend-like annotations
    legend1 = ax1.text(0.02, 0.98,
                       'inferno: dark=low, bright=high',
                       transform=ax1.transAxes, va='top', ha='left', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.3, edgecolor='white'))
    legend2 = ax2.text(0.02, 0.98,
                       'magma: bright = older locked regions',
                       transform=ax2.transAxes, va='top', ha='left', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.3, edgecolor='white'))

    def animate(_):
        uni.evolve()
        # Compute active % for diagnostics
        stress = (np.abs(np.roll(uni.state, 1, axis=0) - uni.state) +
                  np.abs(np.roll(uni.state, 1, axis=1) - uni.state))
        active_pct = 100.0 * np.sum(stress > uni.yield_point) / uni.state.size

        ax1.clear()
        im1 = ax1.imshow(uni.state, cmap='inferno', norm=norm_state)
        ax1.set_title(f"Energy/State (t={uni.time_step})\nBudget: {uni.global_energy:.1f} | Active: {active_pct:.1f}%")
        ax1.set_axis_off()

        ax2.clear()
        im2 = ax2.imshow(uni.hierarchy_level, cmap='magma', norm=norm_hier)
        max_hier = uni.hierarchy_level.max()
        ax2.set_title(f"Emergent Hierarchy (Stability Map)\nBright = Locked | Max age: {max_hier:.0f}")
        ax2.set_axis_off()
        # Re-add legend texts after clear
        ax1.add_artist(legend1)
        ax2.add_artist(legend2)
        return [im1, im2]

    anim = FuncAnimation(fig, animate, interval=interval, frames=frames, blit=False)
    if live:
        plt.show()
    else:
        try:
            writer = PillowWriter(fps=max(1, int(1000/interval)))
            anim.save(outfile, writer=writer)
            print(f"Saved animation: {outfile}")
        except Exception as e:
            print(f"GIF save failed ({e}); saving first frame to mrrc_genesis.png")
            fig.savefig('mrrc_genesis.png', dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='MRRC-Genesis: energy-led emergence with locking and hierarchy')
    ap.add_argument('--size', type=int, default=DEFAULT_GRID_SIZE)
    ap.add_argument('--initial-energy', type=float, default=DEFAULT_INITIAL_ENERGY)
    ap.add_argument('--dissipation-rate', type=float, default=DEFAULT_DISSIPATION_RATE)
    ap.add_argument('--yield-point', type=float, default=DEFAULT_YIELD_POINT)
    ap.add_argument('--coupling', type=float, default=DEFAULT_COUPLING)
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--interval', type=int, default=50)
    ap.add_argument('--seed', type=int, default=None, help='RNG seed (None=random each run)')
    ap.add_argument('--outfile', type=str, default='mrrc_genesis.gif')
    ap.add_argument('--live', action='store_true')
    args = ap.parse_args()

    run_sim(size=args.size,
            initial_energy=args.initial_energy,
            dissipation_rate=args.dissipation_rate,
            yield_point=args.yield_point,
            coupling=args.coupling,
            frames=args.frames,
            interval=args.interval,
            seed=args.seed,
            live=args.live,
            outfile=args.outfile)

if __name__ == '__main__':
    main()
