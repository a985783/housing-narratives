"""
P2-2 Diagnostics: Verify Saiz Interaction Sign and Mechanism
==============================================================
1. Sanity Check: Elasticity vs Inelastic (sign should flip)
2. Match Bias Audit: Matched vs Unmatched characteristics
3. Variance Compression Check: SD(volume_growth) by Saiz tercile
4. Full specification with BOTH inventory and Saiz interactions
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import warnings
import re

warnings.filterwarnings('ignore')

# Import Saiz data from 07_mechanism_saiz.py
from importlib.machinery import SourceFileLoader
saiz_module = SourceFileLoader("saiz", "code/07_mechanism_saiz.py").load_module()
SAIZ_ELASTICITY = saiz_module.SAIZ_ELASTICITY
match_saiz = saiz_module.match_saiz

# Config
DATA_PATH = "data/processed/panel_data_real.csv"
CW_PATH = "data/mappings/metro_dma_crosswalk_deterministic.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    cw = pd.read_csv(CW_PATH)
    df = df.merge(cw[['Metro', 'DMA_Code']], left_on='region', right_on='Metro', how='inner')
    df = df.rename(columns={'DMA_Code': 'dma_code', 'd_ln_volume': 'volume_growth', 
                            'n_buy_std': 'n_buy', 'n_risk_std': 'n_risk'})
    df['saiz_elasticity'] = df['region'].apply(match_saiz)
    return df

def diagnostic_1_sign_flip():
    """Check if using Elasticity (instead of Inelastic) flips the interaction sign."""
    print("="*60)
    print("DIAGNOSTIC 1: Sign Flip Check (Elasticity vs Inelastic)")
    print("="*60)
    
    df = load_data()
    df = df.dropna(subset=['n_buy', 'volume_growth', 'saiz_elasticity'])
    df = df[df['inventory'] > 0]
    
    df['quarter'] = pd.to_datetime(df['quarter'])
    df = df.sort_values(['region', 'quarter'])
    df['n_buy_lag'] = df.groupby('region')['n_buy'].shift(1)
    
    # Version A: Inelastic = -Elasticity (current)
    df['Inelastic_std'] = (-df['saiz_elasticity'] - (-df['saiz_elasticity']).mean()) / (-df['saiz_elasticity']).std()
    df['Inter_Inelastic'] = df['n_buy_lag'] * df['Inelastic_std']
    
    # Version B: Elasticity (raw, higher = more elastic)
    df['Elasticity_std'] = (df['saiz_elasticity'] - df['saiz_elasticity'].mean()) / df['saiz_elasticity'].std()
    df['Inter_Elasticity'] = df['n_buy_lag'] * df['Elasticity_std']
    
    df = df.set_index(['region', 'quarter'])
    clusters = df['dma_code']
    
    # Run both
    print("\nUsing Inelastic = -SaizElasticity:")
    mod_a = PanelOLS.from_formula('volume_growth ~ n_buy_lag + Inter_Inelastic + EntityEffects + TimeEffects', df, drop_absorbed=True)
    res_a = mod_a.fit(cov_type='clustered', clusters=clusters)
    print(f"  N_buy × Inelastic: {res_a.params['Inter_Inelastic']:.4f} (p={res_a.pvalues['Inter_Inelastic']:.4f})")
    
    print("\nUsing Elasticity = +SaizElasticity:")
    mod_b = PanelOLS.from_formula('volume_growth ~ n_buy_lag + Inter_Elasticity + EntityEffects + TimeEffects', df, drop_absorbed=True)
    res_b = mod_b.fit(cov_type='clustered', clusters=clusters)
    print(f"  N_buy × Elasticity: {res_b.params['Inter_Elasticity']:.4f} (p={res_b.pvalues['Inter_Elasticity']:.4f})")
    
    # Check
    if np.sign(res_a.params['Inter_Inelastic']) != np.sign(res_b.params['Inter_Elasticity']):
        print("\n✓ PASS: Signs flip correctly. Variable definition is consistent.")
    else:
        print("\n✗ FAIL: Signs did NOT flip. Possible bug in variable construction!")

def diagnostic_2_match_bias():
    """Compare characteristics of matched vs unmatched metros."""
    print("\n" + "="*60)
    print("DIAGNOSTIC 2: Match Bias Audit (Matched vs Unmatched)")
    print("="*60)
    
    df = load_data()
    df['saiz_matched'] = df['saiz_elasticity'].notna().astype(int)
    
    # Group by metro
    metro_stats = df.groupby('region').agg({
        'saiz_matched': 'first',
        'homes_sold': 'mean',
        'median_sale_price': 'mean',
        'inventory': 'mean',
        'n_buy': 'mean',
        'volume_growth': 'std'
    }).reset_index()
    
    matched = metro_stats[metro_stats['saiz_matched'] == 1]
    unmatched = metro_stats[metro_stats['saiz_matched'] == 0]
    
    print(f"\nMatched metros: {len(matched)}")
    print(f"Unmatched metros: {len(unmatched)}")
    
    print("\n              | Matched     | Unmatched   | Diff")
    print("-" * 55)
    for col in ['homes_sold', 'median_sale_price', 'inventory', 'n_buy', 'volume_growth']:
        m_mean = matched[col].mean()
        u_mean = unmatched[col].mean()
        diff = m_mean - u_mean
        print(f"{col:15s} | {m_mean:10.2f} | {u_mean:10.2f} | {diff:+.2f}")

def diagnostic_3_variance_compression():
    """Check if volume growth variance is lower in inelastic markets."""
    print("\n" + "="*60)
    print("DIAGNOSTIC 3: Variance Compression by Saiz Tercile")
    print("="*60)
    
    df = load_data()
    df = df.dropna(subset=['saiz_elasticity', 'volume_growth'])
    
    # Create terciles
    tercile_edges = df.groupby('region')['saiz_elasticity'].first().quantile([0.33, 0.67])
    
    def get_tercile(e):
        if e <= tercile_edges.iloc[0]:
            return 'Inelastic (bottom 1/3)'
        elif e <= tercile_edges.iloc[1]:
            return 'Medium'
        else:
            return 'Elastic (top 1/3)'
    
    df['saiz_tercile'] = df['saiz_elasticity'].apply(get_tercile)
    
    # Variance by tercile
    var_by_tercile = df.groupby('saiz_tercile')['volume_growth'].agg(['std', 'mean', 'count'])
    print("\nVolume Growth Statistics by Saiz Tercile:")
    print(var_by_tercile.round(4))
    
    # Check
    inelastic_sd = df[df['saiz_tercile'] == 'Inelastic (bottom 1/3)']['volume_growth'].std()
    elastic_sd = df[df['saiz_tercile'] == 'Elastic (top 1/3)']['volume_growth'].std()
    print(f"\nRatio (Elastic SD / Inelastic SD): {elastic_sd/inelastic_sd:.3f}")
    if elastic_sd > inelastic_sd:
        print("→ Elastic markets have MORE volume variance (supports variance compression hypothesis)")

def diagnostic_4_full_spec():
    """Run full specification with BOTH inventory and Saiz interactions."""
    print("\n" + "="*60)
    print("DIAGNOSTIC 4: Full Spec with BOTH Inventory & Saiz Interactions")
    print("="*60)
    
    df = load_data()
    df = df.dropna(subset=['n_buy', 'volume_growth', 'saiz_elasticity'])
    df = df[df['inventory'] > 0]
    
    df['quarter'] = pd.to_datetime(df['quarter'])
    df = df.sort_values(['region', 'quarter'])
    df['n_buy_lag'] = df.groupby('region')['n_buy'].shift(1)
    df['vol_growth_lag'] = df.groupby('region')['volume_growth'].shift(1)
    
    # Inventory
    df['ln_inv'] = np.log(df['inventory'])
    df['InvTight'] = -df['ln_inv']
    df['InvTight_std'] = (df['InvTight'] - df['InvTight'].mean()) / df['InvTight'].std()
    df['InvTight_lag'] = df.groupby('region')['InvTight_std'].shift(1)
    df['Inter_Nbuy_InvTight'] = df['n_buy_lag'] * df['InvTight_lag']
    
    # Saiz
    df['Inelastic_std'] = (-df['saiz_elasticity'] - (-df['saiz_elasticity']).mean()) / (-df['saiz_elasticity']).std()
    df['Inter_Nbuy_Saiz'] = df['n_buy_lag'] * df['Inelastic_std']
    
    df = df.set_index(['region', 'quarter'])
    clusters = df['dma_code']
    
    # Full model with BOTH interactions
    print("\nFull Specification (Both Inventory and Saiz Interactions):")
    mod = PanelOLS.from_formula(
        'volume_growth ~ n_buy_lag + InvTight_lag + Inter_Nbuy_InvTight + Inter_Nbuy_Saiz + vol_growth_lag + EntityEffects + TimeEffects', 
        df, drop_absorbed=True
    )
    res = mod.fit(cov_type='clustered', clusters=clusters)
    
    print(res.summary.tables[1])
    
    print("\nInterpretation:")
    inv_phi = res.params['Inter_Nbuy_InvTight']
    saiz_phi = res.params['Inter_Nbuy_Saiz']
    inv_p = res.pvalues['Inter_Nbuy_InvTight']
    saiz_p = res.pvalues['Inter_Nbuy_Saiz']
    
    print(f"  N_buy × InvTight:  {inv_phi:+.4f} (p={inv_p:.3f}) - {'SIG' if inv_p < 0.1 else 'NS'}")
    print(f"  N_buy × Saiz:      {saiz_phi:+.4f} (p={saiz_p:.3f}) - {'SIG' if saiz_p < 0.1 else 'NS'}")

if __name__ == "__main__":
    diagnostic_1_sign_flip()
    diagnostic_2_match_bias()
    diagnostic_3_variance_compression()
    diagnostic_4_full_spec()
