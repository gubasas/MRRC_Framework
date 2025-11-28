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
def _parse_j2000_name(name: str) -> tuple[float, float]:
    """Extract RA/DEC in degrees from J2000 name like J012017+213346."""
    import re
    name = name.strip().upper()
    # Match pattern JHHMMSS+DDMMSS or JHHMMSS-DDMMSS
    match = re.match(r'J(\d{2})(\d{2})(\d{2})([+-])(\d{2})(\d{2})(\d{2})', name)
    if not match:
        return None, None
    hh, mm, ss, sign, dd, dm, ds = match.groups()
    ra_deg = (float(hh) + float(mm)/60.0 + float(ss)/3600.0) * 15.0  # hours to degrees
    dec_deg = float(dd) + float(dm)/60.0 + float(ds)/3600.0
    if sign == '-':
        dec_deg = -dec_deg
    return ra_deg, dec_deg

def parse_webb_quasar_table(path: str) -> pd.DataFrame:
    """
    Parse Webb et al. / King et al. quasar absorption table into expected schema:
    Required output columns: ra_deg, dec_deg, redshift, delta_alpha_over_alpha, error
    
    Handles:
    - Summary tables with RA/DEC columns
    - King et al. format with J2000_name, z_abs, da/a, err
    """
    import re
    
    # Try reading with manual header extraction for King et al. format
    try:
        with open(path) as f:
            header = None
            for line in f:
                if 'J2000_name' in line and 'z_abs' in line:
                    header = re.sub(r'^#\s*', '', line).strip().split()
                    break
            if header:
                df = pd.read_csv(f, sep=r'\s+', names=header, comment='#')
            else:
                f.seek(0)
                df = pd.read_csv(f, sep="\t", comment="#", skipinitialspace=True)
    except Exception:
        try:
            df = pd.read_csv(path, comment="#", skipinitialspace=True)
        except Exception:
            df = pd.read_csv(path, sep=r'\s+', comment="#", skipinitialspace=True)

    cols = {c.lower().strip().replace('_', ''): c for c in df.columns}

    def get_col(*names):
        for n in names:
            n_clean = n.lower().strip().replace('_', '')
            if n_clean in cols:
                return df[cols[n_clean]]
        return None

    def get_col(*names):
        for n in names:
            n_clean = n.lower().strip().replace('_', '')
            if n_clean in cols:
                return df[cols[n_clean]]
        return None

    # Check for J2000 name format (King et al. 2012)
    j2000_name = get_col("j2000name", "j2000", "name", "qso", "quasar")
    z_abs = get_col("zabs", "zabsorption", "z")
    daa = get_col("daa", "deltaalphaoveralpha", "deltaalpha", "da/a")
    err = get_col("err", "error", "sigma", "uncertainty")

    if j2000_name is not None and z_abs is not None and daa is not None and err is not None:
        # King et al. format with J2000 names
        ras, decs = [], []
        for name in j2000_name:
            ra, dec = _parse_j2000_name(str(name))
            ras.append(ra)
            decs.append(dec)
        
        # daa and err are in units of 10^-5
        frac = pd.to_numeric(daa, errors="coerce") * 1e-5
        err_frac = pd.to_numeric(err, errors="coerce") * 1e-5
        
        out = pd.DataFrame({
            "ra_deg": ras,
            "dec_deg": decs,
            "redshift": pd.to_numeric(z_abs, errors="coerce"),
            "delta_alpha_over_alpha": frac,
            "error": err_frac,
            "source": "king2012",
        }).dropna()
        return out

    # Check for explicit RA/DEC columns (summary table format)
    ra = get_col("ra", "radeg", "ra(deg)", "alpha")
    dec = get_col("dec", "decdeg", "dec(deg)", "delta")
    z = get_col("z", "redshift", "zabs", "zabsorption")
    
    if ra is not None and dec is not None and z is not None:
        # Summary table format
        dppm = get_col("deltappm", "deltaalphaoveralphapm", "deltaalphappm", "daappm", "deltaalpha")
        eppm = get_col("errorppm", "sigmappm", "uncertaintyppm", "errppm", "error", "sigma")
        
        if dppm is None or eppm is None:
            raise ValueError("Summary table missing Δα/α or error columns")
        
        # Convert ppm to fraction
        frac = pd.to_numeric(dppm, errors="coerce") * 1e-6
        err_frac = pd.to_numeric(eppm, errors="coerce") * 1e-6
        ra_deg = pd.to_numeric(ra, errors="coerce")
        dec_deg = pd.to_numeric(dec, errors="coerce")
        
        out = pd.DataFrame({
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "redshift": pd.to_numeric(z, errors="coerce"),
            "delta_alpha_over_alpha": frac,
            "error": err_frac,
            "source": "webb_king",
        }).dropna()
        return out
    
    raise ValueError(
        "Could not parse quasar table. Expected either:\n"
        "1. J2000_name, z_abs, da/a, err (King et al. format), or\n"
        "2. RA, DEC, z, Δα/α [ppm], error [ppm] (summary format)"
    )

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
