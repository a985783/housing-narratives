
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import os
import warnings

warnings.filterwarnings('ignore')

# Config
DATA_PATH = "data/processed/panel_data_real.csv"
CW_PATH = "data/mappings/metro_dma_crosswalk_deterministic.csv"  # Updated to deterministic
OUTPUT_DIR = "output"
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Merge Crosswalk
    cw = pd.read_csv(CW_PATH)
    # cw columns: Metro, DMA_Code
    # df column: region
    
    # Merge
    df = df.merge(cw, left_on='region', right_on='Metro', how='inner')
    
    # Rename cols
    mapper = {
        'd_ln_volume': 'volume_growth',
        'n_buy_std': 'n_buy',
        'n_risk_std': 'n_risk',
        'DMA_Code': 'dma_code'
    }
    df = df.rename(columns=mapper)
    
    # Drop missing n_buy (Complete Case)
    df = df.dropna(subset=['n_buy', 'volume_growth', 'inventory'])
    
    print(f"Sample Size: {len(df)}")
    print(f"Metros: {df['region'].nunique()}")
    print(f"DMAs: {df['dma_code'].nunique()}")
    
    return df

def prepare_vars(df):
    df['quarter'] = pd.to_datetime(df['quarter'])
    df = df.set_index(['region', 'quarter']).sort_index()
    
    # Lags
    df['n_buy_lag'] = df.groupby('region')['n_buy'].shift(1)
    df['vol_growth_lag'] = df.groupby('region')['volume_growth'].shift(1)
    
    # Inventory Quartile (Metro Specific)
    # We need to reset index to calc quantile per group
    df = df.reset_index()
    
    def calc_low_inv(g):
        thresh = g['inventory'].quantile(0.25)
        # 1 if inv <= thresh
        return (g['inventory'] <= thresh).astype(int)
        
    df['LowInv'] = df.groupby('region').apply(calc_low_inv).reset_index(level=0, drop=True)
    
    # Interaction
    df['Inter_Nbuy_LowInv'] = df['n_buy_lag'] * df['LowInv']
    
    # Set index back
    df = df.set_index(['region', 'quarter'])
    return df

def run_mechanism(df):
    print("Running Mechanism Regression...")
    
    # Formula
    # volume_growth ~ n_buy_lag + LowInv + n_buy_lag:LowInv + vol_growth_lag + FE
    # Note: LowInv is time-varying, so we keep it.
    
    formula = 'volume_growth ~ n_buy_lag + LowInv + Inter_Nbuy_LowInv + vol_growth_lag + EntityEffects + TimeEffects'
    
    mod = PanelOLS.from_formula(formula, df, drop_absorbed=True)
    res = mod.fit(cov_type='clustered', cluster_entity=True) # Cluster by Entity (Metro) or DMA if possible
    # Given slight sample, robust cluster by Entity is safer? 
    # Or cluster by DMA?
    # df['dma_code'] is column?
    if 'dma_code' in df.columns:
         mod = PanelOLS.from_formula(formula, df, drop_absorbed=True)
         res = mod.fit(cov_type='clustered', clusters=df['dma_code'])
    
    print(res)
    
    # Latex
    with open(os.path.join(TABLES_DIR, "table_mechanism_inventory.tex"), 'w') as f:
        f.write(res.summary.as_latex())
        
    return res

def plot_marginal_effects(res, df):
    # Plot ME of N_buy as function of LowInv (0 or 1)
    # Coefs: n_buy_lag (beta), Inter (phi)
    # ME = beta + phi * LowInv
    
    try:
        beta = res.params['n_buy_lag']
        phi = res.params['Inter_Nbuy_LowInv']
        beta_se = res.std_errors['n_buy_lag']
        phi_se = res.std_errors['Inter_Nbuy_LowInv']
        
        # We need cov(beta, phi) for SE of sum
        # PanelOLS results object has cov params?
        cov_matrix = res.cov
        cov_beta_phi = cov_matrix.loc['n_buy_lag', 'Inter_Nbuy_LowInv']
        
        # Case 0: High Inv (LowInv=0) -> ME = beta
        me_0 = beta
        se_0 = beta_se
        
        # Case 1: Low Inv (LowInv=1) -> ME = beta + phi
        me_1 = beta + phi
        se_1 = np.sqrt(beta_se**2 + phi_se**2 + 2*cov_beta_phi)
        
        # Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        x = [0, 1]
        y = [me_0, me_1]
        yerr = [1.96*se_0, 1.96*se_1]
        
        ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, color='darkblue', linewidth=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['High Inventory\n(Supply Unconstrained)', 'Low Inventory\n(Supply Constrained)'])
        ax.set_ylabel('Marginal Effect of Buy-Side Search\non Volume Growth')
        ax.set_title('Frustrated Demand Mechanism:\nEffect of Search Intensity by Inventory Level')
        ax.axhline(0, color='gray', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "figure_marginal_effect_inventory.pdf"))
        print("Marginal effect plot saved.")
        
    except Exception as e:
        print(f"Plot failed: {e}")

def main():
    df = load_data()
    df = prepare_vars(df)
    res = run_mechanism(df)
    plot_marginal_effects(res, df)

if __name__ == "__main__":
    main()
