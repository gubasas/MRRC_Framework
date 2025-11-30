PY := /usr/local/bin/python3

.PHONY: data analysis quasar clocks cmb scan hooks gitleaks ds ca ca-legacy ca-drive markov life genesis mrrc-doc papers fe clean

data:
	$(PY) fetch_alpha_data.py

analysis:
	$(PY) mrrc_alpha_variation_test.py

quasar:
	$(PY) tests/quasar_dipole.py

clocks:
	$(PY) tests/atomic_clocks.py

cmb:
	$(PY) tests/cbr_cmb.py

scan:
	$(PY) tools/pii_scan.py . || true

hooks:
	pre-commit install || true

gitleaks:
	gitleaks detect --no-git -s . || true

ds:
	detect-secrets scan > .secrets.baseline || true

ca:
	$(PY) simulations/ca_mrrc.py

ca-legacy:
	$(PY) simulations/ca_mrrc.py --no-mode-locked --expand-every 100 --expand-size 8 --drive-amp 0.0

ca-drive:
	$(PY) simulations/ca_mrrc.py --mode-locked --expand-every 0 --drive-amp 1e-2 --drive-period 60 --drive-duty 0.5

markov:
	$(PY) simulations/mrrc_markov.py

life:
	$(PY) simulations/mrrc_life.py

genesis:
	$(PY) simulations/mrrc_genesis.py --outfile mrrc_genesis.gif --frames 300

mrrc-doc:
	command -v pdflatex >/dev/null 2>&1 && pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs docs/What_is_MRRC.tex || echo 'pdflatex not installed; skipping PDF build'

papers:
	@# refresh top-level V3/V4 from supplementary subtree if present
	@[ -f supplementary/MRRC-Framework-V3/MRRC_V3.pdf ] && cp supplementary/MRRC-Framework-V3/MRRC_V3.pdf V3.pdf || true
	@[ -f supplementary/MRRC-Framework-V3/MRRC_V3.tex ] && cp supplementary/MRRC-Framework-V3/MRRC_V3.tex V3.tex || true
	@[ -f supplementary/MRRC-Framework-V3/MRRC_CT_V4.pdf ] && cp supplementary/MRRC-Framework-V3/MRRC_CT_V4.pdf V4.pdf || true

fe:
	$(PY) analysis/fe_ka_latency.py

clean:
	rm -f *.png tests/*.png simulations/*.png analysis/*.png
