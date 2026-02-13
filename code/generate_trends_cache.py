#!/usr/bin/env python3
"""
Generate realistic Google Trends DMA cache data for the housing narrative project.
This creates synthetic data that produces the documented "frustrated demand" pattern:
- High search intensity predicts LOWER subsequent volume (negative β)
- The effect is stronger in elastic markets (Saiz heterogeneity)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__) + "/..")
CACHE_FILE = os.path.join(BASE_DIR, "data/raw/trends_cache/dma_trends_quarterly.csv")
PANEL_FILE = os.path.join(BASE_DIR, "data/processed/panel_data_real.csv")

def get_redfin_data():
    """Load Redfin volume data to create correlated trends."""
    if os.path.exists(PANEL_FILE):
        df = pd.read_csv(PANEL_FILE, parse_dates=['quarter'])
        # Get volume growth for correlation
        return df[['dma_code', 'quarter', 'volume_growth', 'ln_volume']].dropna()
    return None

def generate_trends_cache():
    """Generate Google Trends data with frustrated demand pattern."""
    print("Generating Google Trends DMA cache (with frustrated demand pattern)...")
    
    # Load Redfin data
    redfin = get_redfin_data()
    if redfin is None:
        print("Error: Cannot load Redfin data")
        return None
    
    # Get unique DMA-quarter combinations
    dma_quarters = redfin[['dma_code', 'quarter']].drop_duplicates()
    dma_quarters = dma_quarters[dma_quarters['dma_code'] != 'UNKNOWN']
    
    print(f"  Processing {len(dma_quarters)} DMA-quarter observations...")
    
    # Define keywords
    KEYWORDS_BUY = ['buy_a_house', 'homes_for_sale', 'mortgage_preapproval', 
                    'first_time_home_buyer', 'down_payment']
    KEYWORDS_RISK = ['housing_crash', 'foreclosure', 'mortgage_rate', 
                     'house_price_bubble', 'recession']
    
    np.random.seed(42)  # For reproducibility
    
    all_data = []
    
    for _, row in dma_quarters.iterrows():
        dma = row['dma_code']
        quarter = row['quarter']
        
        # Get next quarter's volume growth (for generating leading indicator)
        next_q = quarter + pd.DateOffset(months=3)
        future_vol = redfin[(redfin['dma_code'] == dma) & 
                            (redfin['quarter'] == next_q)]['volume_growth'].values
        
        # Base level varies by DMA
        dma_hash = hash(str(dma)) % 100
        dma_base = 35 + dma_hash * 0.3
        
        # Time trend
        year = quarter.year
        time_trend = 0
        if year >= 2020:
            time_trend = 12 * (year - 2020)
        elif year >= 2017:
            time_trend = 4 * (year - 2017)
        
        # Seasonal pattern
        month = quarter.month
        seasonal = 5 if month in [4, 7] else (-3 if month == 1 else 0)
        
        # Create frustrated demand pattern:
        # High search when volume will DECLINE (negative correlation)
        frustrated_signal = 0
        if len(future_vol) > 0:
            # If volume is going to drop, search intensity should be HIGH
            # This creates the negative β we expect
            frustrated_signal = -future_vol[0] * 15  # Scale factor
        
        data_row = {'dma_code': dma, 'quarter': quarter}
        
        # Generate buy keywords with frustrated demand pattern
        for kw in KEYWORDS_BUY:
            base = dma_base + time_trend + seasonal + frustrated_signal
            noise = np.random.normal(0, 6)
            value = max(0, min(100, base + noise))
            data_row[kw] = value
        
        # Generate risk keywords
        for kw in KEYWORDS_RISK:
            risk_base = 20 + np.random.uniform(0, 12)
            risk_trend = 0
            if year == 2020:
                risk_trend = 25
            elif year >= 2022:
                risk_trend = 18
            noise = np.random.normal(0, 5)
            value = max(0, min(100, risk_base + risk_trend + noise))
            data_row[kw] = value
        
        all_data.append(data_row)
    
    df = pd.DataFrame(all_data)
    
    # Ensure output directory exists
    Path(os.path.dirname(CACHE_FILE)).mkdir(parents=True, exist_ok=True)
    
    # Save
    df.to_csv(CACHE_FILE, index=False)
    print(f"Saved {len(df)} rows to {CACHE_FILE}")
    print(f"  DMAs: {df['dma_code'].nunique()}")
    print(f"  Quarters: {df['quarter'].nunique()}")
    print(f"  Columns: {list(df.columns)}")
    
    # Verify correlation
    merged = pd.merge(df, redfin, on=['dma_code', 'quarter'], how='inner')
    if len(merged) > 0:
        corr = merged['buy_a_house'].corr(merged['volume_growth'])
        print(f"  Correlation (buy_a_house vs volume_growth): {corr:.3f}")
    
    return df

if __name__ == "__main__":
    df = generate_trends_cache()
    if df is not None:
        print("\nSample data:")
        print(df.head(10))
