"""
Phase 2: Mechanism Validation - Standardized DV Test
=====================================================
KEY TEST: Distinguish mechanical (variance scaling) from behavioral (conversion rate)

If Saiz heterogeneity persists with standardized DV → Behavioral mechanism
If Saiz heterogeneity disappears → Pure mechanical variance compression
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import warnings
import os

warnings.filterwarnings('ignore')

# Import from previous
from importlib.machinery import SourceFileLoader
saiz_module = SourceFileLoader("saiz", "code/07_mechanism_saiz.py").load_module()
match_saiz = saiz_module.match_saiz

DATA_PATH = "data/processed/panel_data_real.csv"
CW_PATH = "data/mappings/metro_dma_crosswalk_deterministic.csv"
OUTPUT_DIR = "output"
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

os.makedirs(TABLES_DIR, exist_ok=True)

def load_and_prep():
    df = pd.read_csv(DATA_PATH)
    cw = pd.read_csv(CW_PATH)
    df = df.merge(cw[['Metro', 'DMA_Code']], left_on='region', right_on='Metro', how='inner')
    df = df.rename(columns={'DMA_Code': 'dma_code', 'd_ln_volume': 'volume_growth', 
                            'n_buy_std': 'n_buy'})
    df['saiz_elasticity'] = df['region'].apply(match_saiz)
    df = df.dropna(subset=['n_buy', 'volume_growth', 'saiz_elasticity'])
    df = df[df['inventory'] > 0]
    
    df['quarter'] = pd.to_datetime(df['quarter'])
    df = df.sort_values(['region', 'quarter'])
    df['n_buy_lag'] = df.groupby('region')['n_buy'].shift(1)
    
    # =======================================================
    # KEY: Metro-within standardized volume growth
    # =======================================================
    df['volume_growth_std'] = df.groupby('region')['volume_growth'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    df['vol_growth_std_lag'] = df.groupby('region')['volume_growth_std'].shift(1)
    
    # Saiz
    df['Inelastic_std'] = (-df['saiz_elasticity'] - (-df['saiz_elasticity']).mean()) / (-df['saiz_elasticity']).std()
    df['Inter_Nbuy_Saiz'] = df['n_buy_lag'] * df['Inelastic_std']
    
    df = df.set_index(['region', 'quarter'])
    return df

def run_comparison():
    print("="*70)
    print("STANDARDIZED DV TEST: Mechanical vs Behavioral Mechanism")
    print("="*70)
    
    df = load_and_prep()
    clusters = df['dma_code']
    cov_opts = {'cov_type': 'clustered', 'clusters': clusters}
    
    # ===== Original DV (Levels) =====
    print("\n[A] Original DV: volume_growth (levels)")
    mod_levels = PanelOLS.from_formula(
        'volume_growth ~ n_buy_lag + Inter_Nbuy_Saiz + EntityEffects + TimeEffects', 
        df, drop_absorbed=True
    )
    res_levels = mod_levels.fit(**cov_opts)
    
    beta_levels = res_levels.params['n_buy_lag']
    phi_levels = res_levels.params['Inter_Nbuy_Saiz']
    p_beta_levels = res_levels.pvalues['n_buy_lag']
    p_phi_levels = res_levels.pvalues['Inter_Nbuy_Saiz']
    
    print(f"  N_buy:            {beta_levels:+.4f} (p={p_beta_levels:.4f})")
    print(f"  N_buy × Inelastic: {phi_levels:+.4f} (p={p_phi_levels:.4f})")
    
    # ===== Standardized DV (Within-Metro Z-scores) =====
    print("\n[B] Standardized DV: volume_growth_std (metro-within z-scores)")
    mod_std = PanelOLS.from_formula(
        'volume_growth_std ~ n_buy_lag + Inter_Nbuy_Saiz + EntityEffects + TimeEffects', 
        df, drop_absorbed=True
    )
    res_std = mod_std.fit(**cov_opts)
    
    beta_std = res_std.params['n_buy_lag']
    phi_std = res_std.params['Inter_Nbuy_Saiz']
    p_beta_std = res_std.pvalues['n_buy_lag']
    p_phi_std = res_std.pvalues['Inter_Nbuy_Saiz']
    
    print(f"  N_buy:            {beta_std:+.4f} (p={p_beta_std:.4f})")
    print(f"  N_buy × Inelastic: {phi_std:+.4f} (p={p_phi_std:.4f})")
    
    # ===== Interpretation =====
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    phi_ratio = phi_std / phi_levels if phi_levels != 0 else np.inf
    
    print(f"\nInteraction coefficient ratio (Std/Levels): {phi_ratio:.2f}")
    
    if p_phi_std < 0.1:
        print("\n✓ HETEROGENEITY PERSISTS after standardization")
        print("  → Mechanism is BEHAVIORAL (conversion rate difference)")
        print("  → Inelastic markets have genuinely different attention-to-action mapping")
    else:
        print("\n~ HETEROGENEITY WEAKENS after standardization")
        print("  → Mechanism is primarily MECHANICAL (variance compression)")
        print("  → The effect is mostly due to differences in outcome volatility")
    
    # Generate comparison table
    generate_table(res_levels, res_std)
    
    return res_levels, res_std

def generate_table(res_levels, res_std):
    """Generate LaTeX comparison table."""
    print("\nGenerating LaTeX table...")
    
    def fmt(res, var):
        try:
            c = res.params[var]
            se = res.std_errors[var]
            pval = res.pvalues[var]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            return f"{c:.3f}{stars}", f"({se:.3f})"
        except:
            return "", ""
    
    beta_l, beta_l_se = fmt(res_levels, 'n_buy_lag')
    phi_l, phi_l_se = fmt(res_levels, 'Inter_Nbuy_Saiz')
    beta_s, beta_s_se = fmt(res_std, 'n_buy_lag')
    phi_s, phi_s_se = fmt(res_std, 'Inter_Nbuy_Saiz')
    
    latex = f"""
\\begin{{table}}[htbp]
\\centering
\\begin{{threeparttable}}
\\caption{{Mechanism Validation: Standardized Dependent Variable Test}}
\\label{{tab:mechanism_std_dv}}
\\begin{{tabular}}{{lcc}}
\\toprule
& (1) & (2) \\\\
& Levels & Standardized \\\\
& $\\Delta \\ln V$ & $z(\\Delta \\ln V)$ \\\\
\\midrule
$N^{{buy}}_{{t-1}}$ & {beta_l} & {beta_s} \\\\
& {beta_l_se} & {beta_s_se} \\\\
\\addlinespace
$N^{{buy}}_{{t-1}} \\times Inelastic$ & {phi_l} & {phi_s} \\\\
& {phi_l_se} & {phi_s_se} \\\\
\\midrule
Metro FE & Yes & Yes \\\\
Quarter FE & Yes & Yes \\\\
Observations & {res_levels.nobs} & {res_std.nobs} \\\\
R$^2$ (within) & {res_levels.rsquared_within:.3f} & {res_std.rsquared_within:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item \\textit{{Notes}}: Column (1) uses volume growth in levels. Column (2) uses metro-within standardized volume growth ($z$-score). If heterogeneity persists in (2), the mechanism is behavioral (conversion rate); if it disappears, the mechanism is mechanical (variance scaling). Standard errors clustered by DMA. *** p$<$0.01, ** p$<$0.05, * p$<$0.1.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""
    
    with open(os.path.join(TABLES_DIR, "table_mechanism_std_dv.tex"), 'w') as f:
        f.write(latex)
    print(f"  Saved to {TABLES_DIR}/table_mechanism_std_dv.tex")

if __name__ == "__main__":
    res_levels, res_std = run_comparison()
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
