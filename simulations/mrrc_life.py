"""
MRRC Life (didactic, hierarchical growth)
- A Game-of-Life-like grid with:
  - Mode-locked clusters (stable/locked domains that don't cost to maintain)
  - Growth events (clusters expand when conditions met)
  - Drive windows (external influence toggles local rules)
  - Hierarchy: cells -> clusters -> super-clusters (3 levels)

Visualizes how systems can grow, lock, and develop under MRRC-style principles.
Saves an animation to mrrc_life.mp4 (if ffmpeg available) or a frame sequence.
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class MRRCGrid:
    def __init__(self, w: int, h: int, seed: int = 123,
                 lock_threshold: int = 4,
                 growth_threshold: int = 5,
                 drive_period: int | None = 50,
                 drive_duty: float = 0.4):
        rng = np.random.default_rng(seed)
        self.w, self.h = w, h
        # State: 0 empty, 1 alive (developing), 2 locked (mode-locked), 3 leader (cluster head)
        self.state = rng.integers(0, 2, size=(h, w), dtype=np.uint8)
        self.lock_threshold = lock_threshold
        self.growth_threshold = growth_threshold
        self.drive_period = drive_period
        self.drive_duty = drive_duty
        self.t = 0
        # Track simple metrics
        self.metrics = {
            'alive': [],
            'locked': [],
            'leaders': [],
        }

    def neighbors_alive(self, y: int, x: int) -> int:
        s = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny = (y + dy) % self.h
                nx = (x + dx) % self.w
                s += 1 if self.state[ny, nx] in (1, 2, 3) else 0
        return s

    def drive_on(self) -> bool:
        if not self.drive_period:
            return False
        phase = self.t % self.drive_period
        return phase < int(self.drive_duty * self.drive_period)

    def step(self):
        self.t += 1
        drive = self.drive_on()
        nxt = self.state.copy()

        # Base rule (GoL-like for alive=1):
        for y in range(self.h):
            for x in range(self.w):
                s = self.neighbors_alive(y, x)
                c = self.state[y, x]
                if c == 0:  # empty
                    if s == 3:
                        nxt[y, x] = 1
                elif c == 1:  # developing
                    if s < 2 or s > 3:
                        nxt[y, x] = 0
                elif c in (2, 3):  # locked or leader: stable by default
                    # locked stays unless strong perturbation
                    pass

        # Locking: cells with high local support become locked (2)
        for y in range(self.h):
            for x in range(self.w):
                s = self.neighbors_alive(y, x)
                if nxt[y, x] == 1 and s >= self.lock_threshold:
                    nxt[y, x] = 2

        # Leaders: in dense locked regions, designate heads (3)
        for y in range(self.h):
            for x in range(self.w):
                if nxt[y, x] == 2:
                    s = self.neighbors_alive(y, x)
                    if s >= self.growth_threshold:
                        nxt[y, x] = 3

        # Growth: leaders promote adjacent empties to developing during drive or periodic pulses
        if drive or (self.drive_period and self.t % max(1, self.drive_period) == 0):
            for y in range(self.h):
                for x in range(self.w):
                    if nxt[y, x] == 3:
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                ny = (y + dy) % self.h
                                nx = (x + dx) % self.w
                                if nxt[ny, nx] == 0:
                                    nxt[ny, nx] = 1

        # Perturbation during drive: a small fraction of locked cells can unlock to developing
        if drive:
            mask = (nxt == 2)
            idx = np.argwhere(mask)
            if idx.size > 0:
                k = max(1, idx.shape[0]//100)
                sel_idx = np.random.choice(idx.shape[0], size=k, replace=False)
                sel = idx[sel_idx]
                for (yy, xx) in sel:
                    nxt[yy, xx] = 1

        self.state = nxt
        # Update metrics
        self.metrics['alive'].append(int(np.sum(self.state == 1)))
        self.metrics['locked'].append(int(np.sum(self.state == 2)))
        self.metrics['leaders'].append(int(np.sum(self.state == 3)))


def run_animation(width=64, height=64, frames=400, interval=60,
                  lock_threshold=4, growth_threshold=5,
                  drive_period=50, drive_duty=0.4, seed=123,
                  outfile='mrrc_life.gif', live=False):
        grid = MRRCGrid(width, height, seed=seed,
                         lock_threshold=lock_threshold,
                         growth_threshold=growth_threshold,
                         drive_period=drive_period,
                         drive_duty=drive_duty)

        # Use a distinct ListedColormap for states
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#101010', '#3BA3EC', '#6CCB5F', '#E27D60'])
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(grid.state, vmin=0, vmax=3, cmap=cmap, interpolation='nearest')
        title = ax.set_title('MRRC Life: grow, lock, develop (illustrative)')
        ax.set_axis_off()
        # Legend-like annotation
        legend_txt = ax.text(0.02, 0.98,
                             '0 empty\n1 developing\n2 locked\n3 leader',
                             transform=ax.transAxes,
                             va='top', ha='left', fontsize=9,
                             bbox=dict(boxstyle='round', facecolor='black', alpha=0.3, edgecolor='white'))
        drive_txt = ax.text(0.98, 0.02, '', transform=ax.transAxes, va='bottom', ha='right', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

        def update(_):
            grid.step()
            im.set_data(grid.state)
            # Update drive indicator
            drive_txt.set_text('DRIVE' if grid.drive_on() else '')
            # Update title with simple metrics
            if grid.t % 5 == 0:
                a = grid.metrics['alive'][-1] if grid.metrics['alive'] else 0
                l = grid.metrics['locked'][-1] if grid.metrics['locked'] else 0
                h = grid.metrics['leaders'][-1] if grid.metrics['leaders'] else 0
                title.set_text(f'MRRC Life (t={grid.t}) — dev={a} lock={l} lead={h}')
            return (im, title, legend_txt, drive_txt)

        anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
        if live:
            plt.show()
        else:
            try:
                writer = PillowWriter(fps=max(1, int(1000/interval)))
                anim.save(outfile, writer=writer)
                print(f'Saved animation: {outfile}')
            except Exception as e:
                print(f'GIF save failed ({e}); saving first frame to mrrc_life.png')
                fig.savefig('mrrc_life.png', dpi=200)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='MRRC Life: hierarchical growth and locking (didactic)')
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--height', type=int, default=64)
    ap.add_argument('--frames', type=int, default=400)
    ap.add_argument('--interval', type=int, default=60)
    ap.add_argument('--lock-th', type=int, default=4)
    ap.add_argument('--grow-th', type=int, default=5)
    ap.add_argument('--drive-period', type=int, default=50)
    ap.add_argument('--drive-duty', type=float, default=0.4)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--outfile', type=str, default='mrrc_life.gif')
    ap.add_argument('--live', action='store_true', help='show live animation window')
    args = ap.parse_args()

    run_animation(width=args.width, height=args.height, frames=args.frames, interval=args.interval,
                  lock_threshold=args.lock_th, growth_threshold=args.grow_th,
                  drive_period=args.drive_period, drive_duty=args.drive_duty, seed=args.seed,
                  outfile=args.outfile, live=args.live)


if __name__ == '__main__':
    main()
