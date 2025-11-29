PY := /usr/local/bin/python3

.PHONY: data analysis quasar clocks cmb scan hooks gitleaks ds ca ca-legacy ca-drive markov fe clean

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

fe:
	$(PY) analysis/fe_ka_latency.py

clean:
	rm -f *.png tests/*.png simulations/*.png analysis/*.png
