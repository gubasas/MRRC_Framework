import os
import sys
import requests
from alpha_variation_data.ingestion import parse_webb_quasar_table, parse_clock_constraints_table
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "alpha_variation_data")

# Known public dataset URLs
KING_2012_TABLE_A1_URL = "https://academic.oup.com/mnras/article-lookup/doi/10.1111/j.1365-2966.2012.20852.x"  # Fallback: manual download

def download(url: str, out_path: str):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"[OK] Saved: {out_path}")

def main():
    quasar_url = os.getenv("QUASAR_CATALOG_URL")
    clock_url = os.getenv("CLOCK_CONSTRAINTS_URL")
    auto_download = os.getenv("AUTO_DOWNLOAD_KING2012", "").lower() == "true"
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if auto_download:
        print("[INFO] AUTO_DOWNLOAD_KING2012 enabled. Attempting to fetch King et al. 2012 table...")
        print("[WARN] Direct URL not available. Please manually download Table A1 from:")
        print("       https://academic.oup.com/mnras/article/422/4/3370/1050330")
        print("       Or check arXiv ancillary files at https://arxiv.org/abs/1202.4758")
        print("       Then run: QUASAR_CATALOG_URL=file:///path/to/tableA1.dat python3 tools/download_datasets.py")
    
    if quasar_url:
        q_tmp = os.path.join(OUT_DIR, "_raw_quasar.tmp")
        if quasar_url.startswith("file://"):
            import shutil
            shutil.copy(quasar_url[7:], q_tmp)
        else:
            download(quasar_url, q_tmp)
        # Parse via adapter
        df = parse_webb_quasar_table(q_tmp)
        q_out = os.path.join(OUT_DIR, "alpha_variation_quasar_data.csv")
        df.to_csv(q_out, index=False)
        os.remove(q_tmp)
        print(f"[OK] Parsed and saved: {q_out} ({len(df)} rows)")
    
    if clock_url:
        c_tmp = os.path.join(OUT_DIR, "_raw_clock.tmp")
        if clock_url.startswith("file://"):
            import shutil
            shutil.copy(clock_url[7:], c_tmp)
        else:
            download(clock_url, c_tmp)
        df = parse_clock_constraints_table(c_tmp)
        c_out = os.path.join(OUT_DIR, "alpha_variation_clock_data.csv")
        df.to_csv(c_out, index=False)
        os.remove(c_tmp)
        print(f"[OK] Parsed and saved: {c_out} ({len(df)} rows)")
    
    if not quasar_url and not clock_url and not auto_download:
        print("[ERROR] Set QUASAR_CATALOG_URL and/or CLOCK_CONSTRAINTS_URL to downloadable CSV/DAT URLs.")
        print("        Or set AUTO_DOWNLOAD_KING2012=true for instructions.")
        sys.exit(1)
    
    print("Done.")

if __name__ == "__main__":
    main()
