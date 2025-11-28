import os
import sys
import requests

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "alpha_variation_data")

def download(url: str, out_path: str):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"[OK] Saved: {out_path}")

def main():
    quasar_url = os.getenv("QUASAR_CATALOG_URL")
    clock_url = os.getenv("CLOCK_CONSTRAINTS_URL")
    if not quasar_url and not clock_url:
        print("[ERROR] Set QUASAR_CATALOG_URL and/or CLOCK_CONSTRAINTS_URL to downloadable CSV URLs.")
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    if quasar_url:
        q_out = os.path.join(OUT_DIR, "alpha_variation_quasar_data.csv")
        download(quasar_url, q_out)
    if clock_url:
        c_out = os.path.join(OUT_DIR, "alpha_variation_clock_data.csv")
        download(clock_url, c_out)
    print("Done.")

if __name__ == "__main__":
    main()
