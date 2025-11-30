# MRRC_Framework

MRRC V5.0 hypothesis tests and simulations.

Contents:
- `mrrc_alpha_variation_test.py`: end-to-end α-variation analysis and plotting
- `fetch_alpha_data.py`: dataset generator/fetcher (quasars, clocks, pulsars)
- `alpha_variation_data/`: CSVs + metadata used by tests
- `tests/quasar_dipole.py`: quasar dipole fit and plot
- `tests/atomic_clocks.py`: atomic clock Φ/c² linear fit and plot
- `simulations/ca_mrrc.py`: Toy CA illustrating MRRC cost concepts (didactic, not physical)
- `simulations/mrrc_markov.py`: MRRC Markov toy with mode-locked accounting, β·Φ coupling, drive/expansion (didactic)
- `simulations/mrrc_life.py`: MRRC Life — GoL-style, hierarchical grow/lock/develop animation (didactic)
 - `simulations/mrrc_genesis.py`: MRRC-Genesis — energy/yield/dissipation emergence + hierarchy age (didactic)
- `analysis/fe_ka_latency.py`: Fe Kα latency demo (busy-substrate index)
- `mrrc_alpha_variation_report.txt`: generated analysis report
- `mrrc_alpha_variation_analysis.png`: generated summary figure

Quick start:
```bash
python3 fetch_alpha_data.py
python3 mrrc_alpha_variation_test.py
```

Run individual tests:
```bash
python3 tests/quasar_dipole.py
python3 tests/atomic_clocks.py
python3 tests/cbr_cmb.py
python3 simulations/ca_mrrc.py
python3 analysis/fe_ka_latency.py
```

Hypotheses:
- Gravitational potential shifts α via volume_scale (Δα/α ~ β·Φ/c²)
- Rotation modulates π term (Δα/α ~ γ·(ω/ω_c)²)
- Busy-substrate index explains latency-like spectral deviations

Notes:
- Scripts save figures to PNG and do not block on display.
- Replace paths/commands as needed if running outside macOS system Python.

Atomic clock dataset (provisional drift constraints):

| Clock Pair | Epoch | (dα/α)/dt (yr⁻¹) | 1σ | ΔK_α | Reference |
|------------|-------|------------------|-----|-------|-----------|
| Al⁺/Hg⁺ | 2000–2007 | -1.6×10⁻¹⁷ | 2.3×10⁻¹⁷ | 3.2 | Rosenband et al. Science 319 (2008) |
| Yb⁺ (E3/E2) | 2009–2013 | -0.7×10⁻¹⁷ | 2.1×10⁻¹⁷ | 7.0 | Godun et al. PRL 113 (2014) |
| Dysprosium | 2013 | -5.8×10⁻¹⁷ | 6.9×10⁻¹⁷ | ~1×10⁸ | Leefer et al. PRL 111 (2013) |
| Sr/Cs | 2012 | -5.5×10⁻¹⁵ | 4.8×10⁻¹⁵ | 0.06 | Guéna et al. PRL 109 (2012) |
| Hg⁺/Sr | 2007 | 0.0 | 2.3×10⁻¹⁶ | 1.3 | Fortier et al. PRL 98 (2007) |
| Yb⁺/Sr | 2016 | 4.0×10⁻¹⁸ | 8.0×10⁻¹⁸ | 5.0 | Nemitz et al. Nat Photonics 10 (2016) |

CSV: `alpha_variation_data/alpha_variation_clock_data.csv` contains these rows. Annual modulation amplitudes were not included (not all primary sources report a resolved seasonal signal); potential-based β fits currently exclude pure drift rows.

Review source: Ludlow et al. (arXiv:1407.0164) used for consolidated sensitivity coefficients ΔK_α. Replace provisional numbers with direct extraction if higher precision updates are added.

Docs walkthroughs:
```bash
python3 docs/alpha_variation_walkthrough.py
```
Open `docs/alpha_variation_walkthrough.ipynb` in VS Code to run cells.

Privacy & security hygiene:
- Use relative paths in docs/code (no home directories).
- Do not commit virtualenvs; use `.venv/` in `.gitignore` (already set).
- Scan for PII/secrets before publishing:
	- `make scan` (lightweight patterns)
	- Optional: run `gitleaks` or `detect-secrets` for deeper scans
- If sensitive info ever landed in Git history, rewrite using `git filter-repo` or BFG and force-push.

Make targets (optional):
```bash
make analysis   # main α-variation pipeline
make cmb        # CMB consistency test (report + plot)
make scan       # PII/secrets scan (best-effort)
make ca         # CA toy (mode-locked + drive + expansion)
make ca-legacy  # CA with legacy pay-to-maintain behavior
make ca-drive   # CA with active drive only (no expansion)
make markov     # Markov toy (entropy, dS/dt, costs)
make life       # MRRC Life animation (mp4 or fallback png)
make genesis    # MRRC-Genesis (yield + dissipation emergent locking)
make mrrc-doc   # Build What_is_MRRC.tex to PDF (if pdflatex present)
```

Note on simulations: The CA is a visualization aid. It separates “maintenance” vs “chargeable change” costs and lets you toggle weak-field coupling, drive, and expansion. It is not a physical simulation of MRRC dynamics; parameters (e.g., weak-field flips) are visually exaggerated to make effects apparent.
Genesis adds an energy budget, yield threshold, dissipation tax, and hierarchy aging; Life shows hierarchical locking/growth with leaders; Markov focuses on entropy trajectories. All are illustrative lenses on MRRC principles, not predictive physics engines.

CI artifacts:
- On push to `main`, GitHub Actions uploads `mrrc_alpha_variation_report.txt` and `mrrc_alpha_variation_analysis.png` to the workflow run artifacts.
