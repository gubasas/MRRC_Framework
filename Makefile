PY := /usr/local/bin/python3

.PHONY: data analysis quasar clocks ca fe clean

data:
	$(PY) fetch_alpha_data.py

analysis:
	$(PY) mrrc_alpha_variation_test.py

quasar:
	$(PY) tests/quasar_dipole.py

clocks:
	$(PY) tests/atomic_clocks.py

ca:
	$(PY) simulations/ca_mrrc.py

fe:
	$(PY) analysis/fe_ka_latency.py

clean:
	rm -f *.png tests/*.png simulations/*.png analysis/*.png
