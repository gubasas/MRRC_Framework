import os
import sys
import pandas as pd

REQ_QUASAR_COLS = {"ra_deg", "dec_deg", "redshift", "delta_alpha_over_alpha", "error"}
REQ_CLOCK_COLS = {"delta_phi_over_c2", "delta_alpha_over_alpha", "error"}

def check_file(path: str) -> bool:
    if not path:
        print("[ERROR] Path is empty.")
        return False
    if not os.path.isabs(path):
        print(f"[ERROR] Path must be absolute: {path}")
        return False
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return False
    return True

def validate_quasar(path: str) -> bool:
    if not check_file(path):
        return False
    df = pd.read_csv(path)
    cols = set(df.columns)
    missing = REQ_QUASAR_COLS - cols
    if missing:
        print(f"[ERROR] Quasar CSV missing columns: {sorted(missing)}")
        print("Expected columns:", sorted(REQ_QUASAR_COLS))
        return False
    print(f"[OK] Quasar CSV valid with {len(df)} rows.")
    return True

def validate_clocks(path: str) -> bool:
    if not check_file(path):
        return False
    df = pd.read_csv(path)
    cols = set(df.columns)
    # Allow alternate name 'gravitational_potential_ratio'
    alt_cols = set(cols)
    if "gravitational_potential_ratio" in alt_cols:
        alt_cols.add("delta_phi_over_c2")
    missing = REQ_CLOCK_COLS - alt_cols
    if missing:
        print(f"[ERROR] Clock CSV missing columns: {sorted(missing)}")
        print("Expected columns:", sorted(REQ_CLOCK_COLS), "(or 'gravitational_potential_ratio' instead of 'delta_phi_over_c2')")
        return False
    print(f"[OK] Clock CSV valid with {len(df)} rows.")
    return True

def main():
    q = os.getenv("QUASAR_CATALOG_PATH")
    c = os.getenv("CLOCK_CATALOG_PATH")
    ok = True
    if not q:
        print("[ERROR] QUASAR_CATALOG_PATH env var not set.")
        ok = False
    else:
        ok = validate_quasar(q) and ok
    if not c:
        print("[ERROR] CLOCK_CATALOG_PATH env var not set.")
        ok = False
    else:
        ok = validate_clocks(c) and ok
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
