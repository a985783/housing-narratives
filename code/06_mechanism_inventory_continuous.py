"""
P2-1: Mechanism Test - Continuous Inventory Tightness
======================================================
Tests the "Frustrated Demand" hypothesis using:
  InvTight = -log(months_of_supply)
  
Expected: φ < 0 (tighter market → stronger negative N_buy effect)

Outputs:
  - Table with progressive specifications
  - Marginal effect plot (ME of N_buy vs Inventory Tightness)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import warnings
import os

warnings.filterwarnings('ignore')

# Config
DATA_PATH = "data/processed/panel_data_real.csv"
CW_PATH = "data/mappings/metro_dma_crosswalk_deterministic.csv"
OUTPUT_DIR = "output"
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    cw = pd.read_csv(CW_PATH)
    
    # Merge
    df = df.merge(cw[['Metro', 'DMA_Code']], left_on='region', right_on='Metro', how='inner')
    df = df.rename(columns={'DMA_Code': 'dma_code', 'd_ln_volume': 'volume_growth', 'n_buy_std': 'n_buy', 'n_risk_std': 'n_risk'})
    
    # Complete case
    df = df.dropna(subset=['n_buy', 'volume_growth', 'inventory'])
    df = df[df['inventory'] > 0]  # Need positive for log
    
    print(f"Sample: N={len(df)}, Metros={df['region'].nunique()}, DMAs={df['dma_code'].nunique()}")
    return df

def prepare_vars(df):
    df['quarter'] = pd.to_datetime(df['quarter'])
    df = df.sort_values(['region', 'quarter'])
    
    # Lags
    df['n_buy_lag'] = df.groupby('region')['n_buy'].shift(1)
    df['n_risk_lag'] = df.groupby('region')['n_risk'].shift(1)
    df['vol_growth_lag'] = df.groupby('region')['volume_growth'].shift(1)
    
    # =======================================================
    # KEY: InvTight = -log(months_of_supply)
    # Higher = Tighter market (less inventory relative to sales)
    # =======================================================
    df['ln_inv'] = np.log(df['inventory'])
    df['InvTight'] = -df['ln_inv']  # Negate so higher = tighter
    
    # Standardize for interpretability
    df['InvTight_std'] = (df['InvTight'] - df['InvTight'].mean()) / df['InvTight'].std()
    df['InvTight_lag'] = df.groupby('region')['InvTight_std'].shift(1)
    
    # Interaction
    df['Inter_Nbuy_InvTight'] = df['n_buy_lag'] * df['InvTight_lag']
    
    # Set index
    df = df.set_index(['region', 'quarter'])
    return df

def run_mechanism(df):
    print("\nRunning Mechanism Regressions (Continuous Inventory)...")
    results = {}
    
    clusters = df['dma_code']
    cov_opts = {'cov_type': 'clustered', 'clusters': clusters}
    
    # Model 1: N_buy only (baseline)
    print("  M1: N_buy only")
    mod1 = PanelOLS.from_formula('volume_growth ~ n_buy_lag + EntityEffects + TimeEffects', df, drop_absorbed=True)
    res1 = mod1.fit(**cov_opts)
    results['M1'] = res1
    
    # Model 2: + InvTight
    print("  M2: + InvTight")
    mod2 = PanelOLS.from_formula('volume_growth ~ n_buy_lag + InvTight_lag + EntityEffects + TimeEffects', df, drop_absorbed=True)
    res2 = mod2.fit(**cov_opts)
    results['M2'] = res2
    
    # Model 3: + Interaction (THE KEY MODEL)
    print("  M3: + N_buy × InvTight")
    mod3 = PanelOLS.from_formula('volume_growth ~ n_buy_lag + InvTight_lag + Inter_Nbuy_InvTight + EntityEffects + TimeEffects', df, drop_absorbed=True)
    res3 = mod3.fit(**cov_opts)
    results['M3'] = res3
    print(res3)
    
    # Model 4: + Momentum control
    print("  M4: + Momentum")
    mod4 = PanelOLS.from_formula('volume_growth ~ n_buy_lag + InvTight_lag + Inter_Nbuy_InvTight + vol_growth_lag + EntityEffects + TimeEffects', df, drop_absorbed=True)
    res4 = mod4.fit(**cov_opts)
    results['M4'] = res4
    
    return results

def plot_marginal_effects(res, df):
    """
    Plot: Marginal Effect of N_buy across Inventory Tightness distribution.
    ME(InvTight) = β + φ × InvTight
    """
    print("\nGenerating Marginal Effects Plot...")
    
    try:
        beta = res.params['n_buy_lag']
        phi = res.params['Inter_Nbuy_InvTight']
        beta_se = res.std_errors['n_buy_lag']
        phi_se = res.std_errors['Inter_Nbuy_InvTight']
        
        # Get covariance
        cov_matrix = res.cov
        cov_beta_phi = cov_matrix.loc['n_buy_lag', 'Inter_Nbuy_InvTight']
        
        # Range of InvTight (standardized, so approx -2 to +2)
        inv_range = np.linspace(-2, 2, 100)
        
        # Marginal Effect
        me = beta + phi * inv_range
        
        # Standard Error of ME: sqrt(Var(β) + x²Var(φ) + 2x·Cov(β,φ))
        me_se = np.sqrt(beta_se**2 + (inv_range**2) * phi_se**2 + 2 * inv_range * cov_beta_phi)
        
        # 95% CI
        me_lower = me - 1.96 * me_se
        me_upper = me + 1.96 * me_se
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(inv_range, me, 'b-', linewidth=2, label='Marginal Effect')
        ax.fill_between(inv_range, me_lower, me_upper, alpha=0.2, color='blue', label='95% CI')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.axvline(0, color='gray', linestyle=':', linewidth=0.5)
        
        # Labels
        ax.set_xlabel('Inventory Tightness (standardized)\n← Loose Market | Tight Market →', fontsize=11)
        ax.set_ylabel(r'Marginal Effect of $N^{buy}_{t-1}$ on $\Delta \ln V_t$', fontsize=11)
        ax.set_title('Frustrated Demand Mechanism:\nEffect of Search Intensity by Market Tightness', fontsize=12)
        
        # Annotations
        ax.annotate(f'β = {beta:.3f}', xy=(0, beta), xytext=(0.5, beta + 0.02),
                   fontsize=10, ha='left')
        ax.annotate(f'φ = {phi:.3f}', xy=(1.5, beta + phi*1.5), 
                   xytext=(1.5, beta + phi*1.5 - 0.02),
                   fontsize=10, ha='left')
        
        # Legend
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "figure_mechanism_marginal_effect.pdf"), dpi=150)
        plt.savefig(os.path.join(FIGURES_DIR, "figure_mechanism_marginal_effect.png"), dpi=150)
        print(f"  Saved to {FIGURES_DIR}/figure_mechanism_marginal_effect.pdf")
        
    except Exception as e:
        print(f"  Plot failed: {e}")

def generate_latex_table(results):
    """Generate LaTeX table for mechanism results."""
    print("\nGenerating LaTeX Table...")
    
    def fmt(res, var):
        try:
            c = res.params[var]
            se = res.std_errors[var]
            pval = res.pvalues[var]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            return f"{c:.3f}{stars}", f"({se:.3f})"
        except:
            return "", ""
    
    # Extract coefficients
    m1_nbuy, m1_nbuy_se = fmt(results['M1'], 'n_buy_lag')
    m2_nbuy, m2_nbuy_se = fmt(results['M2'], 'n_buy_lag')
    m2_inv, m2_inv_se = fmt(results['M2'], 'InvTight_lag')
    m3_nbuy, m3_nbuy_se = fmt(results['M3'], 'n_buy_lag')
    m3_inv, m3_inv_se = fmt(results['M3'], 'InvTight_lag')
    m3_inter, m3_inter_se = fmt(results['M3'], 'Inter_Nbuy_InvTight')
    m4_nbuy, m4_nbuy_se = fmt(results['M4'], 'n_buy_lag')
    m4_inv, m4_inv_se = fmt(results['M4'], 'InvTight_lag')
    m4_inter, m4_inter_se = fmt(results['M4'], 'Inter_Nbuy_InvTight')
    m4_mom, m4_mom_se = fmt(results['M4'], 'vol_growth_lag')
    
    latex = f"""
\\begin{{table}}[htbp]
\\centering
\\begin{{threeparttable}}
\\caption{{Mechanism Test: Inventory Tightness and Frustrated Demand}}
\\label{{tab:mechanism_inventory}}
\\begin{{tabular}}{{lcccc}}
\\toprule
& (1) & (2) & (3) & (4) \\\\
& Baseline & +Inventory & +Interaction & +Momentum \\\\
\\midrule
$N^{{buy}}_{{t-1}}$ & {m1_nbuy} & {m2_nbuy} & {m3_nbuy} & {m4_nbuy} \\\\
& {m1_nbuy_se} & {m2_nbuy_se} & {m3_nbuy_se} & {m4_nbuy_se} \\\\
\\addlinespace
$InvTight_{{t-1}}$ & & {m2_inv} & {m3_inv} & {m4_inv} \\\\
& & {m2_inv_se} & {m3_inv_se} & {m4_inv_se} \\\\
\\addlinespace
$N^{{buy}}_{{t-1}} \\times InvTight_{{t-1}}$ & & & {m3_inter} & {m4_inter} \\\\
& & & {m3_inter_se} & {m4_inter_se} \\\\
\\addlinespace
$\\Delta \\ln V_{{t-1}}$ & & & & {m4_mom} \\\\
& & & & {m4_mom_se} \\\\
\\midrule
Metro FE & Yes & Yes & Yes & Yes \\\\
Quarter FE & Yes & Yes & Yes & Yes \\\\
Observations & {results['M1'].nobs} & {results['M2'].nobs} & {results['M3'].nobs} & {results['M4'].nobs} \\\\
R$^2$ (within) & {results['M1'].rsquared_within:.3f} & {results['M2'].rsquared_within:.3f} & {results['M3'].rsquared_within:.3f} & {results['M4'].rsquared_within:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item \\textit{{Notes}}: $InvTight = -\\log(months\\_of\\_supply)$ (higher = tighter market). Standard errors clustered by DMA in parentheses. *** p$<$0.01, ** p$<$0.05, * p$<$0.1.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""
    
    with open(os.path.join(TABLES_DIR, "table_mechanism_inventory.tex"), 'w') as f:
        f.write(latex)
    print(f"  Saved to {TABLES_DIR}/table_mechanism_inventory.tex")

def main():
    df = load_data()
    df = prepare_vars(df)
    results = run_mechanism(df)
    plot_marginal_effects(results['M3'], df)
    generate_latex_table(results)
    
    # Summary
    print("\n" + "="*60)
    print("MECHANISM TEST SUMMARY (Continuous Inventory)")
    print("="*60)
    m3 = results['M3']
    print(f"  N_buy (main):      {m3.params['n_buy_lag']:.4f} (p={m3.pvalues['n_buy_lag']:.3f})")
    print(f"  InvTight:          {m3.params['InvTight_lag']:.4f} (p={m3.pvalues['InvTight_lag']:.3f})")
    print(f"  N_buy × InvTight:  {m3.params['Inter_Nbuy_InvTight']:.4f} (p={m3.pvalues['Inter_Nbuy_InvTight']:.3f})")
    print("="*60)
    
    # Interpretation
    phi = m3.params['Inter_Nbuy_InvTight']
    if phi < 0 and m3.pvalues['Inter_Nbuy_InvTight'] < 0.1:
        print("\n✓ MECHANISM CONFIRMED: Tighter markets amplify frustrated demand effect.")
    elif phi < 0:
        print("\n~ Expected sign (φ<0) but not significant. May need larger sample or alternative measure.")
    else:
        print("\n✗ Unexpected sign (φ>0). Consider alternative mechanism or measurement issues.")

if __name__ == "__main__":
    main()
