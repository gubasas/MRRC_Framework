from mrrc_alpha_variation_test import MRRCAnalysis
from alpha_variation_data import ingestion

def main():
    print("Running MRRC α variation analysis walkthrough...")
    analysis = MRRCAnalysis()
    analysis.run_all()
    print("Report saved to mrrc_alpha_variation_report.txt")
    print("Plot saved to mrrc_alpha_variation_analysis.png")

    # Show swapping loaders
    q = ingestion.load_quasar_catalog()
    c = ingestion.load_atomic_clock_constraints()
    p = ingestion.load_pulsar_targets()
    print(f"Loaded quasar rows: {len(q)}, clocks: {len(c)}, pulsars: {len(p)}")
    print("To use real catalogs, replace CSVs referenced in ingestion.py with published datasets.")

if __name__ == "__main__":
    main()
