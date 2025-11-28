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

# Adapters for common published formats
def parse_webb_quasar_table(path: str) -> pd.DataFrame:
    """
    Parse Webb et al. quasar absorption table (TSV/CSV) into expected schema:
    Required output columns: ra_deg, dec_deg, redshift, delta_alpha_over_alpha, error
    Assumes input columns include RA/DEC (deg or sexagesimal), z, delta_ppm, error_ppm.
    """
    # Try TSV first, then CSV
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path)

    cols = {c.lower().strip(): c for c in df.columns}

    def get_col(*names):
        for n in names:
            if n in cols:
                return df[cols[n]]
        raise KeyError(f"Missing required column among: {names}")

    # RA/DEC may be in degrees or sexagesimal; assume degrees if numeric
    ra = get_col("ra", "ra_deg", "ra (deg)")
    dec = get_col("dec", "dec_deg", "dec (deg)")
    z = get_col("z", "redshift")
    dppm = get_col("delta_ppm", "deltaalphaoveralpha_ppm", "delta_alpha_ppm", "da_a_ppm")
    eppm = get_col("error_ppm", "sigma_ppm", "uncertainty_ppm", "err_ppm")

    # Convert ppm to fraction
    frac = pd.to_numeric(dppm, errors="coerce") * 1e-6
    err = pd.to_numeric(eppm, errors="coerce") * 1e-6
    ra_deg = pd.to_numeric(ra, errors="coerce")
    dec_deg = pd.to_numeric(dec, errors="coerce")

    out = pd.DataFrame({
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "redshift": pd.to_numeric(z, errors="coerce"),
        "delta_alpha_over_alpha": frac,
        "error": err,
        "source": "webb2011",
    }).dropna()
    return out

def parse_clock_constraints_table(path: str) -> pd.DataFrame:
    """
    Parse optical clock constraints table into expected schema.
    Expects columns that can map to delta_phi_over_c2 (or gravitational_potential_ratio),
    delta_alpha_over_alpha, and error.
    """
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    def get_col(*names):
        for n in names:
            if n in cols:
                return df[cols[n]]
        raise KeyError(f"Missing required column among: {names}")
    # Accept either potential ratio or explicit delta_phi_over_c2
    phi = get_col("delta_phi_over_c2", "gravitational_potential_ratio", "phi_over_c2")
    daa = get_col("delta_alpha_over_alpha", "delta_alpha", "daa_fraction")
    err = get_col("error", "sigma", "uncertainty")
    out = pd.DataFrame({
        "delta_phi_over_c2": pd.to_numeric(phi, errors="coerce"),
        "delta_alpha_over_alpha": pd.to_numeric(daa, errors="coerce"),
        "error": pd.to_numeric(err, errors="coerce"),
        "source": "clock_constraints",
    }).dropna()
    return out
