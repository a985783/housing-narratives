# Master script to run all analyses
"""
Run all analyses in sequence for the paper replication.

This script executes the full replication pipeline:
1. Data pipeline (if needed)
2. Main analysis
3. Mechanism tests
4. Robustness checks
5. Table generation

Usage:
    python code/run_all.py
"""

import subprocess
import sys
from pathlib import Path

# Configuration
DATA_FILE = Path("data/processed/panel_data_real.csv")
USE_CACHED_DATA = True  # Set to False to refresh all data

def run_script(script_name):
    """Run a Python script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)
    
    result = subprocess.run(
        [sys.executable, f"code/{script_name}"],
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"❌ Error running {script_name}")
        return False
    
    print(f"✅ Completed: {script_name}")
    return True

def main():
    """Main execution pipeline."""
    print("\n" + "="*60)
    print("THE FRICTION GATE: Full Replication Pipeline")
    print("="*60)
    print("\nThis will run all analyses for the paper.\n")
    
    # Check if data exists
    if not DATA_FILE.exists():
        print("⚠️  Processed data not found. Running data pipeline...")
        USE_CACHED_DATA = False
    
    scripts = []
    
    # Step 1: Data pipeline (if needed)
    if not USE_CACHED_DATA:
        scripts.append("03_real_data_pipeline.py")
    
    # Step 2-6: Analyses
    scripts.extend([
        "04_analysis_real.py",          # Main baseline analysis
        "05_interaction_models.py",     # Mechanism: interactions
        "06_discriminant_validity.py",  # Robustness: discriminant validity
        "07_robustness.py",             # Robustness: alternative specs
        "07_mechanism_saiz.py",         # Mechanism: Saiz elasticity
        "11_wild_bootstrap.py",         # Statistical inference
        "12_final_tables.py",           # Generate LaTeX tables
    ])
    
    # Run all scripts
    success_count = 0
    for script in scripts:
        if run_script(script):
            success_count += 1
        else:
            print(f"\n⚠️  Stopping pipeline due to error in {script}")
            break
    
    # Summary
    print("\n" + "="*60)
    print("REPLICATION COMPLETE")
    print("="*60)
    print(f"\n✅ Successfully ran {success_count}/{len(scripts)} scripts")
    print(f"\n📁 Output files available in: output/")
    print(f"📊 Figures: output/figures/")
    print(f"📋 Tables: output/tables/")
    print("\nNext steps:")
    print("  1. Check output/regression_results_real.txt for results")
    print("  2. Compile LaTeX tables from output/tables/")
    print("  3. Compare results to paper Tables 2-4")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
