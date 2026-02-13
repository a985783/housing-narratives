
import re
import pandas as pd
import os

SCRIPT_PATH = "code/03_real_data_pipeline.py"
DATA_PATH = "data/processed/panel_data_real.csv"
OUTPUT_CSV = "data/mappings/metro_dma_crosswalk_main_208_recovered.csv"

def extract_dict():
    with open(SCRIPT_PATH, 'r') as f:
        content = f.read()
    
    # Regex to capture the dict
    # Starts with METRO_DMA_CROSSWALK = {
    # Ends with }
    match = re.search(r"METRO_DMA_CROSSWALK = \{(.*?)\}", content, re.DOTALL)
    if match:
        dict_str = "{" + match.group(1) + "}"
        # Evaluate
        # We need to ensure string quotes are valid
        try:
            d = eval(dict_str)
            return d
        except Exception as e:
            print(f"Eval failed: {e}")
            return None
    return None

def get_dma_for_metro(metro_name, cw_dict):
    for pattern, dma_code in cw_dict.items():
        if pattern.lower() in metro_name.lower():
            return dma_code
    return None

def main():
    print("Extracting Dictionary from 03...")
    cw_dict = extract_dict()
    if not cw_dict:
        print("Failed to extract dictionary.")
        return
        
    print(f"Dictionary size: {len(cw_dict)} keys.")
    
    # Run Match
    df = pd.read_csv(DATA_PATH)
    df['dma_code'] = df['region'].apply(lambda x: get_dma_for_metro(x, cw_dict))
    
    df_mapped = df.dropna(subset=['dma_code'])
    unique_metros = df_mapped['region'].unique()
    print(f"Unique Metros Mapped: {len(unique_metros)}")
    
    if len(unique_metros) > 100:
        # Save
        df_cw = df_mapped[['region', 'dma_code']].drop_duplicates()
        df_cw.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved recovered crosswalk to {OUTPUT_CSV}")
    else:
        print("Still only ~90 matches. The dictionary in 03 is also small.")

if __name__ == "__main__":
    main()
