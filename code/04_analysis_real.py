"""
ECR-CAS Housing Cycles - REAL Data Analysis (Phase 1.8B - DETERMINISTIC CROSSWALK)
==================================================================================
This script runs the empirical analysis using the newly rebuilt deterministic crosswalk.
Key Features:
1. REPRODUCIBLE SAMPLE: Loads from metro_dma_crosswalk_deterministic.csv
2. NO IMPUTATION: Uses complete case analysis (drops missing narratives).
3. DMA CLUSTERING: Clusters standard errors by DMA.
4. VERIFICATION: Prints exact Metros and Clusters counts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import warnings
import os

warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/panel_data_real.csv")
CROSSWALK_PATH = os.path.join(BASE_DIR, "data/mappings/metro_dma_crosswalk_deterministic.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_and_clean_data():
    print("Loading real panel data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run pipeline 03 first.")
    if not os.path.exists(CROSSWALK_PATH):
        raise FileNotFoundError(f"{CROSSWALK_PATH} not found. Run rebuild_crosswalk.py first.")
        
    df = pd.read_csv(DATA_PATH)
    if 'dma_code' in df.columns:
        df = df.drop(columns=['dma_code'])
        
    cw = pd.read_csv(CROSSWALK_PATH)
    
    # Merge crosswalk
    print(f"  Loaded panel: {len(df)} rows")
    print(f"  Loaded crosswalk: {len(cw)} metros")
    
    df = df.merge(cw[['Metro', 'DMA_Code']], left_on='region', right_on='Metro', how='inner')
    df = df.rename(columns={'DMA_Code': 'dma_code'})
    
    print(f"  After crosswalk merge: {len(df)} rows")
    
    # 1. Rename columns to match model spec
    rename_map = {}
    if 'd_ln_volume' in df.columns and 'volume_growth' not in df.columns:
        rename_map['d_ln_volume'] = 'volume_growth'
    if 'n_buy_std' in df.columns and 'n_buy' not in df.columns:
        rename_map['n_buy_std'] = 'n_buy'
    if 'n_risk_std' in df.columns and 'n_risk' not in df.columns:
        rename_map['n_risk_std'] = 'n_risk'
    if 'jumbo_exposure_std' in df.columns and 'jumbo_exposure' not in df.columns:
        rename_map['jumbo_exposure_std'] = 'jumbo_exposure'

    if rename_map:
        df = df.rename(columns=rename_map)
    
    # 2. COMPLETE CASE ANALYSIS (NO IMPUTATION)
    n_total = len(df)
    
    # Drop rows where n_buy is NaN (missing trends)
    df = df.dropna(subset=['n_buy', 'volume_growth'])
    
    n_complete = len(df)
    
    print(f"  Filtering for Complete Cases:")
    print(f"    Total obs after merge: {n_total}")
    print(f"    Complete case obs: {n_complete}")
    print(f"    Dropped (missing narratives): {n_total - n_complete} ({(n_total - n_complete)/n_total:.1%})")
    
    # Verification Stats
    n_metros = df['region'].nunique()
    n_dmas = df['dma_code'].nunique()
    
    print('\n' + '='*50)
    print("PHASE 1.8B SAMPLE VERIFICATION (DETERMINISTIC CW)")
    print('='*50)
    print(f"  Observation Count:      {n_complete}")
    print(f"  Unique Metros (Entities): {n_metros}")
    print(f"  Unique DMAs (Clusters):   {n_dmas}")
    print('='*50 + '\n')
    
    # 3. DMA AGGREGATION (Risk 2 Fix)
    print("\n   [Risk 2 Fix] Aggregating Metros to DMA Level...")
    
    # Ensure dma_code is present
    if 'dma_code' not in df.columns:
        raise ValueError("dma_code missing after merge")

    # Ensure quarter is datetime
    df['quarter'] = pd.to_datetime(df['quarter'])

    # Define aggregation rules
    
    # Helper for weighted average
    def weighted_avg(x, w):
        try:
            return np.average(x, weights=w)
        except ZeroDivisionError:
            return x.mean()

    # Groupby DMA + Quarter
    # Use sum for homes_sold and inventory
    # Use weighted avg for price
    # Use mean for narrative (constant within ID)
    
    dma_groups = df.groupby(['dma_code', 'quarter'])
    
    df_dma = pd.DataFrame()
    df_dma['homes_sold'] = dma_groups['homes_sold'].sum()
    df_dma['inventory'] = dma_groups['inventory'].sum()
    
    # Weighted averages
    # Note: apply is slow, but sample is small (99 DMAs x 50 Q = 5000 groups)
    df_dma['median_sale_price'] = df.groupby(['dma_code', 'quarter']).apply(
        lambda x: np.average(x['median_sale_price'], weights=x['homes_sold'])
    )
    
    # Narratives (Constant within DMA, so mean/first is fine)
    df_dma['n_buy'] = dma_groups['n_buy'].mean()
    df_dma['n_risk'] = dma_groups['n_risk'].mean()
    
    if 'jumbo_exposure' in df.columns:
        df_dma['jumbo_exposure'] = df.groupby(['dma_code', 'quarter']).apply(
            lambda x: np.average(x['jumbo_exposure'], weights=x['homes_sold'])
        )
        
    df_dma = df_dma.reset_index()
    
    print(f"   Aggregated to {len(df_dma)} DMA-Quarter observations.")
    
    # Re-calculate Growth Rates at DMA Level
    df_dma = df_dma.sort_values(['dma_code', 'quarter'])
    
    df_dma['ln_volume'] = np.log(df_dma['homes_sold'].clip(lower=1))
    df_dma['volume_growth'] = df_dma.groupby('dma_code')['ln_volume'].diff()
    
    df_dma['ln_price'] = np.log(df_dma['median_sale_price'].clip(lower=1))
    df_dma['price_growth'] = df_dma.groupby('dma_code')['ln_price'].diff()
    
    df_dma['ln_inventory'] = np.log(df_dma['inventory'].replace(0, np.nan).ffill().bfill())
    
    # Create Lags
    df_dma['n_buy_lag'] = df_dma.groupby('dma_code')['n_buy'].shift(1)
    df_dma['n_risk_lag'] = df_dma.groupby('dma_code')['n_risk'].shift(1)
    df_dma['volume_growth_lag'] = df_dma.groupby('dma_code')['volume_growth'].shift(1)
    df_dma['ln_inventory_lag'] = df_dma.groupby('dma_code')['ln_inventory'].shift(1)
    
    # Prepare Final Panel
    df_dma = df_dma.dropna(subset=['volume_growth', 'n_buy_lag'])
    df_dma = df_dma.set_index(['dma_code', 'quarter'])
    
    # Update Stats
    n_dmas = df_dma.index.get_level_values('dma_code').nunique()
    print(f"   Final DMA Panel: {n_dmas} DMAs, {len(df_dma)} Obs.")
    
    return df_dma

def run_regressions(df):
    print("\nRunning Panel Regressions (DMA Clustering)...")
    results = {}
    
    # Clustering Setting
    cluster_opts = {}
    if 'dma_code' in df.columns or 'dma_code' in df.index.names:
        print(f"  Clustering by Entity (DMA).")
        # If dma_code is index, cluster_entity=True clusters by DMA
        cluster_opts = {'cov_type': 'clustered', 'cluster_entity': True}
    else:
        print("  Warning: Clustering by Entity (Metro) as fallback.")
        cluster_opts = {'cov_type': 'clustered', 'cluster_entity': True}
    
    # Model 1: N_buy only
    print("  Model 1: N_buy")
    mod1 = PanelOLS.from_formula('volume_growth ~ n_buy_lag + EntityEffects + TimeEffects', df, drop_absorbed=True, check_rank=False)
    res1 = mod1.fit(**cluster_opts)
    results['M1'] = res1
    
    # Model 2: N_buy + N_risk
    print("  Model 2: N_buy + N_risk")
    mod2 = PanelOLS.from_formula('volume_growth ~ n_buy_lag + n_risk_lag + EntityEffects + TimeEffects', df, drop_absorbed=True, check_rank=False)
    res2 = mod2.fit(**cluster_opts)
    results['M2'] = res2
    
    # Model 3: Controls (Added Inventory)
    print("  Model 3: Controls (+Inventory)")
    mod3 = PanelOLS.from_formula('volume_growth ~ n_buy_lag + n_risk_lag + volume_growth_lag + ln_inventory_lag + EntityEffects + TimeEffects', df, drop_absorbed=True, check_rank=False)
    res3 = mod3.fit(**cluster_opts)
    results['M3'] = res3
    print(res3)
    
    # Model 6: Price (for Volume vs Price comparison)
    print("  Model 6: Price Effect")
    mod6 = PanelOLS.from_formula('price_growth ~ n_buy_lag + n_risk_lag + ln_inventory_lag + EntityEffects + TimeEffects', df, drop_absorbed=True)
    res6 = mod6.fit(**cluster_opts)
    results['M6_Price'] = res6
    
    # Save Summary
    with open(os.path.join(OUTPUT_DIR, "regression_results_real.txt"), 'w') as f:
        f.write(str(res3))
        
    return results

def generate_latex_table(results, df):
    print("\nGenerating LaTeX Table...")
    def get_coef(res, name):
        try:
            c = res.params[name]
            se = res.std_errors[name]
            pval = res.pvalues[name]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            return f"{c:.3f}{stars}", f"({se:.3f})"
        except:
            return "", ""
            
    m1_c, m1_se = get_coef(results['M1'], 'n_buy_lag')
    m2_c_buy, m2_se_buy = get_coef(results['M2'], 'n_buy_lag')
    m2_c_risk, m2_se_risk = get_coef(results['M2'], 'n_risk_lag')
    m3_c_buy, m3_se_buy = get_coef(results['M3'], 'n_buy_lag')
    m3_c_risk, m3_se_risk = get_coef(results['M3'], 'n_risk_lag')
    
    n_entities = df.index.get_level_values(0).nunique()
    n_time = df.index.get_level_values(1).nunique()

    latex = f"""
\\begin{{table}}[htbp]
\\centering
\\begin{{threeparttable}}
\\caption{{Results with Deterministic Crosswalk (Phase 1.8B)}}
\\label{{tab:real_results}}
\\begin{{tabular}}{{lccc}}
\\toprule
& (1) & (2) & (3) \\\\
\\midrule
$N^{{buy}}_{{t-1}}$ & {m1_c} & {m2_c_buy} & {m3_c_buy} \\\\
& {m1_se} & {m2_se_buy} & {m3_se_buy} \\\\
\\addlinespace
$N^{{risk}}_{{t-1}}$ & & {m2_c_risk} & {m3_c_risk} \\\\
& & {m2_se_risk} & {m3_se_risk} \\\\
\\midrule
Inventory & No & No & Yes \\\\
DMA FE & Yes & Yes & Yes \\\\
Quarter FE & Yes & Yes & Yes \\\\
Observations & {results['M1'].nobs} & {results['M2'].nobs} & {results['M3'].nobs} \\\\
R-squared & {results['M1'].rsquared_within:.3f} & {results['M2'].rsquared_within:.3f} & {results['M3'].rsquared_within:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item \\textit{{Notes}}: DMA-level analysis ({n_entities} DMAs). Dependent variable is volume growth aggregated to DMA level. Standard errors clustered by DMA.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""
    with open(os.path.join(TABLES_DIR, "table_real.tex"), 'w') as f:
        f.write(latex)
        
    print(f"  Saved to {os.path.join(TABLES_DIR, 'table_real.tex')}")

def main():
    df = load_and_clean_data()
    results = run_regressions(df)
    generate_latex_table(results, df)
    
    # Print Key comparison
    m3 = results['M3']
    m6 = results['M6_Price']
    if 'n_buy_lag' in m3.params and 'n_buy_lag' in m6.params:
        vol_eff = m3.params['n_buy_lag']
        price_eff = m6.params['n_buy_lag']
        print("\nKEY FINDING (Deterministic Crosswalk):")
        print(f"  Volume Effect: {vol_eff:.4f}")
        print(f"  Price Effect:  {price_eff:.4f}")
        if price_eff != 0:
            print(f"  Ratio (V/P):   {vol_eff/price_eff:.2f}x")

if __name__ == "__main__":
    main()
