import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "")

def load_quasar_catalog(path: str | None = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, "alpha_variation_quasar_data.csv")
    df = pd.read_csv(path)
    df["source"] = df.get("source", "simulated")
    return df

def load_atomic_clock_constraints(path: str | None = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, "alpha_variation_clock_data.csv")
    df = pd.read_csv(path)
    df["source"] = df.get("source", "simulated")
    return df

def load_pulsar_targets(path: str | None = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, "alpha_variation_pulsar_data.csv")
    df = pd.read_csv(path)
    df["source"] = df.get("source", "simulated")
    return df

# Notes: replace CSVs with real catalogs
PROVENANCE_NOTES = {
    "quasar": "Replace with SDSS/VLT/Keck α-variation catalog; include RA/Dec, z, Δα/α, uncertainties, references.",
    "clocks": "Replace with NIST/PTB published constraints (Al+/Hg+, Yb/Sr); include epoch, Δα/α bounds, setup.",
    "pulsar": "Replace with ATNF or X-ray line measurements relevant to α; include spin, environment params.",
}
