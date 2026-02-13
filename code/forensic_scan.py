#!/usr/bin/env python3
"""
Phase 1.8A: Forensic Scan for dma_code Column
Scans all CSV/Parquet/Pickle files in the project for a 'dma_code' column.
"""
import os
import pandas as pd
from pathlib import Path

SCAN_DIRS = ['data', 'output', 'cache', '.']
EXTENSIONS = ['.csv', '.parquet', '.pkl', '.pickle', '.tsv']

def scan_file(filepath):
    """Try to read file and check for dma_code column."""
    ext = filepath.suffix.lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(filepath, nrows=5)
        elif ext == '.tsv':
            df = pd.read_csv(filepath, sep='\t', nrows=5)
        elif ext == '.parquet':
            df = pd.read_parquet(filepath)
        elif ext in ['.pkl', '.pickle']:
            df = pd.read_pickle(filepath)
        else:
            return None
            
        cols = df.columns.tolist()
        # Check for dma_code or similar
        dma_cols = [c for c in cols if 'dma' in c.lower() or 'geo' in c.lower() or c.startswith('US-')]
        if dma_cols:
            return {
                'file': str(filepath),
                'columns': cols,
                'dma_related': dma_cols,
                'shape': df.shape if hasattr(df, 'shape') else 'unknown'
            }
    except Exception as e:
        pass
    return None

def main():
    print("="*60)
    print("Phase 1.8A: Forensic Scan for dma_code Evidence")
    print("="*60)
    
    found = []
    
    for scan_dir in SCAN_DIRS:
        if not os.path.exists(scan_dir):
            continue
        for root, dirs, files in os.walk(scan_dir):
            # Skip venv, .git, node_modules
            dirs[:] = [d for d in dirs if d not in ['venv', '.git', 'node_modules', '__pycache__']]
            for f in files:
                fp = Path(root) / f
                if fp.suffix.lower() in EXTENSIONS:
                    result = scan_file(fp)
                    if result:
                        found.append(result)
    
    if found:
        print(f"\n✅ FOUND {len(found)} files with DMA-related columns:\n")
        for item in found:
            print(f"  📁 {item['file']}")
            print(f"     DMA Cols: {item['dma_related']}")
            print(f"     Shape: {item['shape']}")
            print()
    else:
        print("\n❌ No files with dma_code or geo columns found.")
        print("   The evidence file may have been deleted or never saved.")
    
    # Also check for Google Trends cache files
    print("\n" + "="*60)
    print("Checking for Google Trends Cache (may contain geo codes)...")
    print("="*60)
    
    cache_dirs = ['data/cache', 'cache', 'data/raw']
    for cd in cache_dirs:
        if os.path.exists(cd):
            files = list(Path(cd).glob('*'))
            print(f"\n{cd}: {len(files)} files")
            for f in files[:10]:
                print(f"  - {f.name}")

if __name__ == "__main__":
    main()
