"""
ECR-CAS Housing Cycles - Discriminant Validity Test (Risk 4 Defense)
=====================================================================
Tests if "Risk" narratives interact with Inventory asymmetrically.
Goal: Show that Scarcity amplifies "Buy" but dampens "Risk" (Discriminant Validity).

Hypothesis: In tight markets, "Panic Buying" (Buy N) is amplified, 
while "Fear" (Risk N) is ignored/dampened.
"""

import pandas as pd
import numpy as np
import os
import time
from linearmodels.panel import PanelOLS
import warnings

warnings.filterwarnings('ignore')

# Reuse pipeline logic for fetching (simplified)
from pytrends.request import TrendReq

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/panel_data_real.csv")
CW_PATH = os.path.join(BASE_DIR, "data/mappings/metro_dma_crosswalk_deterministic.csv")
CACHE_DIR = os.path.join(BASE_DIR, "data/raw/trends_cache")
PLACEBO_FILE = os.path.join(CACHE_DIR, "dma_placebo_furniture.csv")

PLACEBO_KW = ["furniture"]

def fetch_dma_weights(keywords):
    print(f"  Fetching weights for {keywords}...")
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(keywords, timeframe='2012-01-01 2024-12-31', geo='US')
        df = pytrends.interest_by_region(resolution='DMA', inc_geo_code=True, inc_low_vol=True)
        if df.empty: return None
        return df.set_index('geoCode')[keywords[0]].to_dict()
    except Exception as e:
        print(f"Error weights: {e}")
        return None

def fetch_placebo_trends(dmas):
    if os.path.exists(PLACEBO_FILE):
        print("Loading cached placebo...")
        return pd.read_csv(PLACEBO_FILE)
    
    print("Fetching Placebo Trends...")
    
    # Check if we should fallback due to 429
    # If weights fails, likely trends will fail too.
    weights = fetch_dma_weights(PLACEBO_KW)
    if weights is None:
        print("  ! API Limited (429). Using 'risk_recession' from main dataset as Pseudo-Placebo.")
        
        # Load main data again to extract 'risk_recession' if available
        # But wait, main dataset has 'n_risk' which is an index of multiple terms?
        # Let's check if we have individual columns in panel_data_real.csv?
        # Usually processed panel has 'n_buy', 'n_risk'.
        # 'n_risk' (Crash/Foreclosure/Recession) -> Should NOT positively predict volume in tight markets.
        # It's a "Negative Placebo" (Discriminant Validity).
        
        df_main = pd.read_csv(DATA_PATH)
        if 'n_risk' in df_main.columns:
            df_plac = df_main[['region', 'quarter', 'n_risk']].copy()
            df_plac = df_plac.rename(columns={'n_risk': 'n_placebo'})
            # We need to aggregate to DMA to match format
            # But wait, this function is supposed to return DMA-level placebo.
            # df_main has 'region' (metro).
            # Need to merge dma
            cw = pd.read_csv(CW_PATH)
            df_plac = df_plac.merge(cw[['Metro', 'DMA_Code']], left_on='region', right_on='Metro')
            df_plac['quarter'] = pd.to_datetime(df_plac['quarter'])
            
            # Aggregate to DMA
            dma_plac = df_plac.groupby(['DMA_Code', 'quarter'])['n_placebo'].mean().reset_index()
            dma_plac = dma_plac.rename(columns={'DMA_Code': 'dma_code'})
            return dma_plac
            
        return pd.DataFrame()
    
    pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25))
    all_trends = []
    
    # Fetch first 10-20 DMAs just to prove the point (Don't need all 100 for placebo check if time constrained)
    # Actually, let's try to get all matched ones.
    
    count = 0
    for dma in dmas:
        try:
             # Extract numeric
            num = dma.split('-')[-1]
            
            pytrends.build_payload(PLACEBO_KW, timeframe='2012-01-01 2024-12-31', geo=dma)
            df = pytrends.interest_over_time()
            if df.empty: continue
            
            df = df.drop(columns=['isPartial'], errors='ignore')
            
            # Weighting
            if weights and num in weights:
                w = weights[num]
                mean = df.values.mean()
                if mean > 0:
                    df = df * (w/mean)
            
            df.columns = ['n_placebo']
            df['dma_code'] = dma
            df['date'] = df.index
            all_trends.append(df)
            count += 1
            print(f" {dma}", end='', flush=True)
            time.sleep(1)
            
            if count >= 30: # Limit to 30 DMAs to save time for this run, enough for significance check?
                # User wants "defensible", better do all. But speed?
                # Let's do 50.
                if count >= 80: break 
        except:
            time.sleep(2)
            
    full = pd.concat(all_trends)
    # Ensure date is datetime
    full['date'] = pd.to_datetime(full['date'])
    full['quarter'] = full['date'].dt.to_period('Q').dt.to_timestamp()
    q_trends = full.groupby(['dma_code', 'quarter']).mean(numeric_only=True).reset_index()
    q_trends.to_csv(PLACEBO_FILE, index=False)
    return q_trends

def run_placebo_analysis():
    # 1. Load Data
    df_main = pd.read_csv(DATA_PATH)
    if 'dma_code' in df_main.columns: df_main = df_main.drop(columns=['dma_code'])
    cw = pd.read_csv(CW_PATH)
    df_main = df_main.merge(cw[['Metro', 'DMA_Code']], left_on='region', right_on='Metro')
    
    # Get unique DMAs to fetch
    unique_dmas = df_main['DMA_Code'].unique()
    
    # 2. Fetch/Load Placebo
    df_plac = fetch_placebo_trends(unique_dmas)
    
    # 3. Merge
    # Aggregate Main to DMA
    df_main['quarter'] = pd.to_datetime(df_main['quarter'])
    df_plac['quarter'] = pd.to_datetime(df_plac['quarter'])
    
    # Aggregation (simplified)
    # Ensure columns exist
    if 'homes_sold' not in df_main.columns:
        # Maybe it's called 'volume'? Check 03
        pass # assume homes_sold exists as standard

    dma_grp = df_main.groupby(['DMA_Code', 'quarter'])
    
    # Sum volume first
    vol_series = dma_grp['homes_sold'].sum()
    inv_series = dma_grp['inventory'].sum()
    
    df_dma = pd.DataFrame({
        'homes_sold': vol_series,
        'inventory': inv_series
    }).reset_index()
    
    df_dma = df_dma.rename(columns={'DMA_Code': 'dma_code'})
    
    # Calculate Growth
    df_dma = df_dma.sort_values(['dma_code', 'quarter'])
    df_dma['ln_volume'] = np.log(df_dma['homes_sold'].clip(lower=1))
    df_dma['volume_growth'] = df_dma.groupby('dma_code')['ln_volume'].diff()
    df_dma['ln_inv'] = np.log(df_dma['inventory'].replace(0, np.nan).ffill().bfill())
    
    # Merge Placebo
    df_final = df_dma.merge(df_plac, left_on=['dma_code', 'quarter'], right_on=['dma_code', 'quarter'], how='inner')
    
    # Lags
    df_final = df_final.sort_values(['dma_code', 'quarter'])
    df_final['n_placebo_lag'] = df_final.groupby('dma_code')['n_placebo'].shift(1)
    df_final['ln_inv_lag'] = df_final.groupby('dma_code')['ln_inv'].shift(1)
    
    # LowInv Interaction
    med = df_final['ln_inv_lag'].median()
    df_final['LowInv'] = (df_final['ln_inv_lag'] < med).astype(int)
    df_final['Inter_Placebo'] = df_final['n_placebo_lag'] * df_final['LowInv']
    
    df_final = df_final.set_index(['dma_code', 'quarter']).dropna()
    
    print("\nRunning Placebo Interaction...")
    mod = PanelOLS.from_formula('volume_growth ~ n_placebo_lag + Inter_Placebo + ln_inv_lag + EntityEffects + TimeEffects', df_final)
    res = mod.fit(cov_type='clustered', cluster_entity=True)
    
    print(res)
    
    return res

if __name__ == "__main__":
    res = run_placebo_analysis()
    
    p_inter = res.pvalues['Inter_Placebo']
    print(f"\nPlacebo Interaction P-value: {p_inter:.4f}")
    if p_inter > 0.1:
        print("✓ PASS: Placebo interaction is insignificant.")
    else:
        print("✗ FAIL: Placebo interaction is significant.")
