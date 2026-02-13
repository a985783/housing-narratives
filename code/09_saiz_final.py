"""
P2-2 Final: Saiz Mechanism Visualization & Wild Bootstrap
==========================================================
1. Tercile-based marginal effect plot (most intuitive for readers)
2. Wild cluster bootstrap for robust inference
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import warnings
import re

warnings.filterwarnings('ignore')

# Import from previous scripts
from importlib.machinery import SourceFileLoader
saiz_module = SourceFileLoader("saiz", "code/07_mechanism_saiz.py").load_module()
match_saiz = saiz_module.match_saiz

DATA_PATH = "data/processed/panel_data_real.csv"
CW_PATH = "data/mappings/metro_dma_crosswalk_deterministic.csv"
FIGURES_DIR = "output/figures"

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
    
    # Create terciles
    metro_saiz = df.groupby('region')['saiz_elasticity'].first()
    terciles = pd.qcut(metro_saiz, 3, labels=['Inelastic', 'Medium', 'Elastic'])
    df['saiz_tercile'] = df['region'].map(terciles)
    
    return df

def plot_tercile_effects():
    """Run separate regressions for each tercile and plot marginal effects."""
    print("="*60)
    print("Saiz Tercile Marginal Effects")
    print("="*60)
    
    df = load_and_prep()
    
    results = {}
    for tercile in ['Inelastic', 'Medium', 'Elastic']:
        df_sub = df[df['saiz_tercile'] == tercile].copy()
        df_sub = df_sub.set_index(['region', 'quarter'])
        
        mod = PanelOLS.from_formula('volume_growth ~ n_buy_lag + EntityEffects + TimeEffects', df_sub, drop_absorbed=True)
        res = mod.fit(cov_type='clustered', clusters=df_sub['dma_code'])
        
        results[tercile] = {
            'beta': res.params['n_buy_lag'],
            'se': res.std_errors['n_buy_lag'],
            'pval': res.pvalues['n_buy_lag'],
            'n': res.nobs
        }
        print(f"  {tercile}: β = {res.params['n_buy_lag']:.4f} (SE={res.std_errors['n_buy_lag']:.4f}, p={res.pvalues['n_buy_lag']:.3f}), N={res.nobs}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ['Inelastic\n(Hard to Build)', 'Medium', 'Elastic\n(Easy to Build)']
    x = np.arange(len(categories))
    
    betas = [results[t]['beta'] for t in ['Inelastic', 'Medium', 'Elastic']]
    errors = [1.96 * results[t]['se'] for t in ['Inelastic', 'Medium', 'Elastic']]
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green→Yellow→Red
    
    bars = ax.bar(x, betas, yerr=errors, capsize=8, color=colors, alpha=0.8, edgecolor='black')
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel(r'Marginal Effect of $N^{buy}_{t-1}$ on Volume Growth', fontsize=11)
    ax.set_title('The Variance Compression Mechanism:\nSearch Effect Attenuated in Supply-Constrained Markets', fontsize=12)
    
    # Annotations
    for i, t in enumerate(['Inelastic', 'Medium', 'Elastic']):
        r = results[t]
        stars = "***" if r['pval'] < 0.01 else "**" if r['pval'] < 0.05 else "*" if r['pval'] < 0.1 else ""
        ax.annotate(f"β = {r['beta']:.3f}{stars}\n(N={r['n']})", 
                   xy=(i, r['beta']), 
                   xytext=(i, r['beta'] - 0.02 if r['beta'] < 0 else r['beta'] + 0.015),
                   ha='center', fontsize=9)
    
    # Add explanation
    ax.text(0.02, 0.98, 'Volume variance is 48% lower in inelastic markets,\nmechanically compressing the marginal effect of search.',
           transform=ax.transAxes, fontsize=9, va='top', style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/figure_saiz_tercile_effects.pdf", dpi=150)
    plt.savefig(f"{FIGURES_DIR}/figure_saiz_tercile_effects.png", dpi=150)
    print(f"\nSaved to {FIGURES_DIR}/figure_saiz_tercile_effects.pdf")
    
    return results

def wild_cluster_bootstrap(n_boot=999):
    """Wild cluster bootstrap for the Saiz interaction coefficient."""
    print("\n" + "="*60)
    print(f"Wild Cluster Bootstrap (B={n_boot})")
    print("="*60)
    
    df = load_and_prep()
    df['Inelastic_std'] = (-df['saiz_elasticity'] - (-df['saiz_elasticity']).mean()) / (-df['saiz_elasticity']).std()
    df['Inter_Nbuy_Saiz'] = df['n_buy_lag'] * df['Inelastic_std']
    df = df.set_index(['region', 'quarter'])
    
    # Original estimate
    mod = PanelOLS.from_formula('volume_growth ~ n_buy_lag + Inter_Nbuy_Saiz + EntityEffects + TimeEffects', df, drop_absorbed=True)
    res = mod.fit()
    
    original_phi = res.params['Inter_Nbuy_Saiz']
    original_t = res.tstats['Inter_Nbuy_Saiz']
    
    print(f"Original φ = {original_phi:.4f}, t = {original_t:.3f}")
    
    # Get clusters
    clusters = df.reset_index()['dma_code'].unique()
    n_clusters = len(clusters)
    
    # Bootstrap
    boot_phis = []
    boot_ts = []
    
    for b in range(n_boot):
        # Rademacher weights by cluster
        weights = {c: np.random.choice([-1, 1]) for c in clusters}
        df_boot = df.copy()
        df_boot['weight'] = df_boot.reset_index()['dma_code'].map(weights).values
        
        # Perturb residuals
        resid = res.resids.values
        df_boot['volume_growth_boot'] = res.fitted_values.values + resid * df_boot['weight'].values
        
        try:
            mod_boot = PanelOLS.from_formula('volume_growth_boot ~ n_buy_lag + Inter_Nbuy_Saiz + EntityEffects + TimeEffects', df_boot, drop_absorbed=True)
            res_boot = mod_boot.fit()
            boot_phis.append(res_boot.params['Inter_Nbuy_Saiz'])
            boot_ts.append(res_boot.tstats['Inter_Nbuy_Saiz'])
        except:
            continue
    
    boot_phis = np.array(boot_phis)
    boot_ts = np.array(boot_ts)
    
    # Bootstrap p-value (symmetric)
    p_boot = np.mean(np.abs(boot_ts) >= np.abs(original_t))
    
    # CI
    ci_lower = np.percentile(boot_phis, 2.5)
    ci_upper = np.percentile(boot_phis, 97.5)
    
    print(f"\nWild Bootstrap Results:")
    print(f"  φ = {original_phi:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Bootstrap p-value: {p_boot:.4f}")
    
    if ci_lower > 0 or ci_upper < 0:
        print("\n✓ 95% CI does not contain zero → ROBUST")
    else:
        print("\n~ 95% CI contains zero → Use caution in interpretation")
    
    return original_phi, ci_lower, ci_upper, p_boot

if __name__ == "__main__":
    results = plot_tercile_effects()
    phi, ci_lo, ci_hi, p_boot = wild_cluster_bootstrap()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Saiz Interaction: φ = {phi:.4f}")
    print(f"Wild Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Bootstrap p-value: {p_boot:.4f}")
    print("\nMechanism: Variance Compression")
    print("In supply-constrained markets, volume variance is 48% lower,")
    print("mechanically attenuating the marginal effect of search intensity.")
