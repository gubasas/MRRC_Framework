# MRRC_Framework

MRRC V5.0 hypothesis tests and simulations.

Contents:
- `mrrc_alpha_variation_test.py`: end-to-end α-variation analysis and plotting
- `fetch_alpha_data.py`: dataset generator/fetcher (quasars, clocks, pulsars)
- `alpha_variation_data/`: CSVs + metadata used by tests
- `tests/quasar_dipole.py`: quasar dipole fit and plot
- `tests/atomic_clocks.py`: atomic clock Φ/c² linear fit and plot
- `simulations/ca_mrrc.py`: MRRC CA recorder simulation (PC1–PC5)
- `analysis/fe_ka_latency.py`: Fe Kα latency demo (busy-substrate index)
- `mrrc_alpha_variation_report.txt`: generated analysis report
- `mrrc_alpha_variation_analysis.png`: generated summary figure

Quick start:
```bash
cd "/Users/alexk/Documents/ GlassMind"
/usr/local/bin/python3 fetch_alpha_data.py
/usr/local/bin/python3 mrrc_alpha_variation_test.py
```

Run individual tests:
```bash
/usr/local/bin/python3 tests/quasar_dipole.py
/usr/local/bin/python3 tests/atomic_clocks.py
/usr/local/bin/python3 simulations/ca_mrrc.py
/usr/local/bin/python3 analysis/fe_ka_latency.py
```

Hypotheses:
- Gravitational potential shifts α via volume_scale (Δα/α ~ β·Φ/c²)
- Rotation modulates π term (Δα/α ~ γ·(ω/ω_c)²)
- Busy-substrate index explains latency-like spectral deviations

Notes:
- Scripts save figures to PNG and do not block on display.
- Replace paths/commands as needed if running outside macOS system Python.

Docs walkthroughs:
```bash
/usr/local/bin/python3 docs/alpha_variation_walkthrough.py
```
Open `docs/alpha_variation_walkthrough.ipynb` in VS Code to run cells.

CI artifacts:
- On push to `main`, GitHub Actions uploads `mrrc_alpha_variation_report.txt` and `mrrc_alpha_variation_analysis.png` to the workflow run artifacts.
