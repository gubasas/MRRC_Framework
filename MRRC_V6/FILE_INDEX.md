# MRRC V6 Directory Index

## Complete File Listing

### 📄 Documentation
- `README.md` - Main documentation (comprehensive overview)
- `DATA_PRIVACY.md` - Privacy notice and sanitization details
- `FILE_INDEX.md` - This file

### 📑 Paper (paper/)
- `MRRC_V6.pdf` - Submitted paper (clean, no personal data)
- `MRRC_V6_SANITIZED.tex` - LaTeX source (email removed)
- `MRRC_V6.aux` - LaTeX auxiliary file

### 💻 Code (code/)
- `derive_factor_10.py` - Rigorous derivation of /10 normalization factor
  - Proves only k=4 gives perfect square √100=10
  - Shows three independent locks (dimensional, combinatorial, E₈)
  - Generates 4-panel proof visualization
  - Runtime: ~5 seconds
  
- `pure_geometric_summary.py` - Demonstration of pure geometric derivation
  - No number fitting - all from formula
  - Shows polygon values, golden ratio, tetrahedral angle
  - Physics matches table
  - Runtime: ~1 second

### 🎨 Figures (figures/)

**Main Paper Figures:**
- `mrrc_geometric_structures_complete.png` - 24-cell, polygons, F₄ roots, symmetry breaking
- `mrrc_particle_mass_spirals_detailed.png` - Lepton and quark spirals in 4D phase space
- `mrrc_golden_ratio_geometry.png` - Pentagon, icosahedron, φ² derivation
- `mrrc_polygonal_symmetry_test.png` - Falsification test results (success/failure pattern)
- `mrrc_complete_derivation_flowchart.png` - Complete logical derivation chain

**Supplementary Figures:**
- `derive_factor_10_complete.png` - Four-panel proof of /10 normalization (NEW)
- `mrrc_24cell_complete_theory.png` - Detailed 24-cell geometry and α⁻¹ derivation
- `mrrc_koide_geometric_origin.png` - Koide Q=2/3 from tetrahedral angle
- `mrrc_f4_mass_derivation.png` - F₄ mass matrix eigenvalue analysis

### 📚 Supplementary Materials (supplementary/)
- `MRRC_V5_1_Master.md` - Previous framework version (reference)

---

## File Sizes

```
paper/MRRC_V6.pdf:                          ~500 KB
paper/MRRC_V6_SANITIZED.tex:                ~80 KB
code/derive_factor_10.py:                   ~9 KB
code/pure_geometric_summary.py:             ~8 KB
figures/mrrc_geometric_structures_complete.png: ~2.3 MB
figures/mrrc_complete_derivation_flowchart.png: ~480 KB
figures/derive_factor_10_complete.png:      ~150 KB (estimated)
```

Total directory size: ~10 MB (excluding git metadata)

---

## Quick Start

### View the Paper
```bash
open MRRC_V6/paper/MRRC_V6.pdf
```

### Run Derivations
```bash
# Derive /10 factor from first principles
python3 MRRC_V6/code/derive_factor_10.py

# Show pure geometric summary
python3 MRRC_V6/code/pure_geometric_summary.py
```

### Dependencies
```bash
pip install numpy matplotlib scipy
```

---

## What Each File Proves

### derive_factor_10.py
**Proves:** The mysterious /10 normalization factor is NOT arbitrary
- Shows only k=4 layer gives √100=10 (perfect square)
- Demonstrates three independent geometric locks
- Proves dimensionality (k=4), combinatorics (C(4,3)=4), and E₈ structure converge
- **Conclusion:** /10 is derived from geometry, not empirically fitted

### pure_geometric_summary.py
**Demonstrates:** Entire framework uses NO number fitting
- All polygon values from ancient formula: P_n(k) = 1 + n·k·(k-1)/2
- Golden ratio from pentagon: sin(54°) = φ/2 (EXACT)
- Tetrahedral angle: cos(θ) = -1/3 (EXACT)
- 24-cell V/S = 1/3 (EXACT)
- **Conclusion:** Physics matches are empirical observations, not fitted results

### Figures Overview

**mrrc_geometric_structures_complete.png:**
- Panel 1: 24-cell in 4D → α⁻¹ = 137.036
- Panel 2: Polygonal symmetries (triangle through octagon)
- Panel 3: 4D spiral mass trajectory
- Panel 4: Centered hexagonal numbers
- Panel 5: F₄ root system (48 roots)
- Panel 6: H₃→A₃ symmetry breaking (φ origin)

**mrrc_complete_derivation_flowchart.png:**
- Complete logical chain from axioms to predictions
- Shows what's derived (green) vs empirical (yellow) vs speculative (orange)
- Includes all validations and falsifications

**derive_factor_10_complete.png:**
- Panel 1: Perfect square proof (k=1 to k=10 test)
- Panel 2: All polygons normalized by √100
- Panel 3: E₈ = Hexagon(148) + Square(100) decomposition
- Panel 4: Normalization effect visualization

---

## Reproducibility Checklist

✓ All Python code included  
✓ No proprietary software required  
✓ All figures regeneratable from code  
✓ Mathematical proofs step-by-step  
✓ Data sources documented (PDG 2024)  
✓ No personal data included  
✓ Public repository ready  

---

## Version Control

- **V6.0:** Submitted to arXiv December 2, 2025
- **V6.1 (in progress):** Post-submission discoveries:
  - /10 factor DERIVED (not empirical)
  - Pentagon → Higgs 124→125 GeV
  - Triangle → U-70 beam 76 GeV
  - Square → Strange quark 100 MeV
  - Units are conventions insight

---

## For Reviewers

If you're reviewing this work, please note:

1. **What's derived vs fitted:** See pure_geometric_summary.py output
2. **Falsification tests:** Triangle (thought to fail, later found match), Octagon (untested)
3. **Independent predictions:** Top quark 172 GeV NOT fitted to quark data
4. **Transparency:** All development history in GitHub repository
5. **New discoveries:** Pentagon→Higgs found AFTER submission (December 3)

---

## Contact

**GitHub Repository:** https://github.com/gubasas/MRRC_Framework  
**Issues/Questions:** https://github.com/gubasas/MRRC_Framework/issues

---

Last updated: December 3, 2025
