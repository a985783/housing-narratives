"""
ECR-CAS Housing Cycles - Robustness Checks (Risk 4)
===================================================
1. Granger Causality (Reverse Prediction)
2. Placebo (If available, but for now just reverse)
3. Inventory Regime Interaction (Optional)

Input: data/processed/panel_data_real.csv
Output: output/tables/table_granger.tex
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import os

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/panel_data_real.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_aggregate():
    print("Loading data for Robustness...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Run 03 pipeline first.")
        
    df = pd.read_csv(DATA_PATH)
    df['quarter'] = pd.to_datetime(df['quarter'])
    
    # Aggregation to DMA (Copying logic from 04 since it's the identification unit)
    if 'dma_code' not in df.columns:
        # If pipeline saved without dma_code or named differently?
        # 03 usually saves dma_code.
        raise ValueError("dma_code missing")
        
    dma_groups = df.groupby(['dma_code', 'quarter'])
    df_dma = pd.DataFrame()
    df_dma['homes_sold'] = dma_groups['homes_sold'].sum()
    
    # Metrics
    df_dma['n_buy'] = dma_groups['n_buy'].mean()
    df_dma['n_risk'] = dma_groups['n_risk'].mean()
    
    df_dma = df_dma.reset_index()
    df_dma = df_dma.sort_values(['dma_code', 'quarter'])
    
    # Growth
    df_dma['ln_volume'] = np.log(df_dma['homes_sold'].clip(lower=1))
    df_dma['volume_growth'] = df_dma.groupby('dma_code')['ln_volume'].diff()
    
    # Lags
    df_dma['n_buy_lag'] = df_dma.groupby('dma_code')['n_buy'].shift(1)
    df_dma['vol_lag'] = df_dma.groupby('dma_code')['volume_growth'].shift(1)
    
    df_dma = df_dma.dropna()
    df_dma = df_dma.set_index(['dma_code', 'quarter'])
    return df_dma

def run_granger(df):
    print("\nRunning Granger Causality Tests...")
    
    # 1. Main Direction: Narrative -> Future Volume
    mod_main = PanelOLS.from_formula('volume_growth ~ n_buy_lag + EntityEffects + TimeEffects', df)
    res_main = mod_main.fit(cov_type='clustered', cluster_entity=True)
    
    # 2. Reverse Direction: Volume -> Future Narrative
    # Does past volume predict current narrative?
    mod_rev = PanelOLS.from_formula('n_buy ~ vol_lag + EntityEffects + TimeEffects', df)
    res_rev = mod_rev.fit(cov_type='clustered', cluster_entity=True)
    
    print("\n--- Granger Causality ---")
    print(f"Narrative -> Volume (t+1): Coef={res_main.params['n_buy_lag']:.3f}, P={res_main.pvalues['n_buy_lag']:.3f}")
    print(f"Volume -> Narrative (t+1): Coef={res_rev.params['vol_lag']:.3f}, P={res_rev.pvalues['vol_lag']:.3f}")
    
    # Save to LaTex
    latex = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Granger Causality / Reverse Prediction}}
\\label{{tab:granger}}
\\begin{{tabular}}{{lcc}}
\\toprule
& (1) Main & (2) Reverse \\\\
& Vol Growth & Narrative \\\\
\\midrule
$N^{{buy}}_{{t-1}}$ & {res_main.params['n_buy_lag']:.3f}{'***' if res_main.pvalues['n_buy_lag']<0.01 else '**' if res_main.pvalues['n_buy_lag']<0.05 else ''} & \\\\
& ({res_main.std_errors['n_buy_lag']:.3f}) & \\\\
\\addlinespace
$VolGrowth_{{t-1}}$ & & {res_rev.params['vol_lag']:.3f}{'***' if res_rev.pvalues['vol_lag']<0.01 else '**' if res_rev.pvalues['vol_lag']<0.05 else ''} \\\\
& & ({res_rev.std_errors['vol_lag']:.3f}) \\\\
\\midrule
Observations & {res_main.nobs} & {res_rev.nobs} \\\\
R-squared & {res_main.rsquared_within:.3f} & {res_rev.rsquared_within:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(os.path.join(OUTPUT_DIR, "table_granger.tex"), 'w') as f:
        f.write(latex)
    print(f"Saved to {os.path.join(OUTPUT_DIR, 'table_granger.tex')}")

if __name__ == "__main__":
    df = load_and_aggregate()
    run_granger(df)
