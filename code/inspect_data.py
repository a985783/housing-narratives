
import pandas as pd

DATA_PATH = "data/processed/panel_data_real.csv"

try:
    df = pd.read_csv(DATA_PATH)
    print("Columns:", df.columns.tolist())
    if 'dma_code' in df.columns:
        print("dma_code found!")
        # Count non-null
        unique_mapped = df[['region', 'dma_code']].dropna().drop_duplicates()
        print(f"Unique Region-DMA Pairs: {len(unique_mapped)}")
        print(unique_mapped.head())
        
        # Save to CSV
        unique_mapped.to_csv("data/mappings/metro_dma_crosswalk_recovered.csv", index=False)
        print("Recovered mapping saved.")
    else:
        print("dma_code NOT found.")
except Exception as e:
    print(f"Error: {e}")
