"""
Phase 2: Wild Cluster Bootstrap (Robust Inference)
===================================================
Fixes shape mismatch by ensuring consistent estimation sample.
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import os
import time

# Import Saiz matching logic
from importlib.machinery import SourceFileLoader
saiz_module = SourceFileLoader("saiz", "code/07_mechanism_saiz.py").load_module()
match_saiz = saiz_module.match_saiz

DATA_PATH = "data/processed/panel_data_real.csv"
CW_PATH = "data/mappings/metro_dma_crosswalk_deterministic.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    cw = pd.read_csv(CW_PATH)
    df = df.merge(cw[['Metro', 'DMA_Code']], left_on='region', right_on='Metro', how='inner')
    df = df.rename(columns={'DMA_Code': 'dma_code', 'd_ln_volume': 'volume_growth', 
                            'n_buy_std': 'n_buy'})
    df['saiz_elasticity'] = df['region'].apply(match_saiz)
    
    df['quarter'] = pd.to_datetime(df['quarter'])
    df = df.sort_values(['region', 'quarter'])
    df['n_buy_lag'] = df.groupby('region')['n_buy'].shift(1)
    
    # Create variable
    df['Inelastic_std'] = (-df['saiz_elasticity'] - (-df['saiz_elasticity']).mean()) / (-df['saiz_elasticity']).std()
    df['Inter_Nbuy_Saiz'] = df['n_buy_lag'] * df['Inelastic_std']
    
    # CRITICAL: Drop NA here to define the EXACT estimation sample
    cols_needed = ['volume_growth', 'n_buy_lag', 'Inter_Nbuy_Saiz', 'dma_code', 'region', 'quarter']
    df = df.dropna(subset=cols_needed)
    df = df[df['inventory'] > 0]
    
    df = df.set_index(['region', 'quarter'])
    return df

def run_wild_bootstrap(B=999):
    print(f"Starting Wild Cluster Bootstrap (B={B})...")
    
    df = load_data()
    n_obs = len(df)
    clusters = df.reset_index()['dma_code'].unique()
    n_clusters = len(clusters)
    print(f"Sample: N={n_obs}, Clusters={n_clusters} (DMAs)")
    
    # 1. Fit Original Model
    print("Fitting original model...")
    mod = PanelOLS.from_formula(
        'volume_growth ~ n_buy_lag + Inter_Nbuy_Saiz + EntityEffects + TimeEffects', 
        df, drop_absorbed=True
    )
    # Use df['dma_code'] directly to preserve index alignment
    res = mod.fit(cov_type='clustered', clusters=df['dma_code'])
    
    original_beta = res.params['Inter_Nbuy_Saiz']
    original_t = res.tstats['Inter_Nbuy_Saiz']
    print(f"Original: φ = {original_beta:.4f}, t = {original_t:.3f}")

    # Store fitted values and residuals
    fitted = res.fitted_values['fitted_values'].values
    resids = res.resids.values
    
    # Map cluster ID to integer index for speed
    # We need dma_code as a series aligned with fitted/resids
    cluster_series = df['dma_code']
    cluster_to_idx = {c: i for i, c in enumerate(clusters)}
    cluster_indices = cluster_series.map(cluster_to_idx).values
    
    # 2. Bootstrap Loop
    boot_t_stats = []
    start_time = time.time()
    
    for b in range(B):
        if b % 100 == 0:
            print(f"  Bootstrap {b}/{B}...")
            
        # Rademacher weights (+1 or -1) for each cluster
        cluster_weights = np.random.choice([1, -1], size=n_clusters)
        obs_weights = cluster_weights[cluster_indices]
        
        y_boot = fitted + resids * obs_weights
        
        df_boot = df.copy()
        df_boot['volume_growth'] = y_boot
        
        try:
            mod_boot = PanelOLS.from_formula(
                'volume_growth ~ n_buy_lag + Inter_Nbuy_Saiz + EntityEffects + TimeEffects', 
                df_boot, drop_absorbed=True
            )
            # Use original clusters for bootstrap fit too
            res_boot = mod_boot.fit(cov_type='clustered', clusters=df['dma_code'])
            
            t_star = (res_boot.params['Inter_Nbuy_Saiz'] - original_beta) / res_boot.std_errors['Inter_Nbuy_Saiz']
            boot_t_stats.append(t_star)
        except Exception as e:
            print(f"  Failed boot {b}: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"Bootstrap finished in {elapsed:.1f}s")
    
    # 3. Inference
    boot_t_stats = np.array(boot_t_stats)
    
    # 95% Confidence Interval for t-stat
    t_lower = np.percentile(boot_t_stats, 2.5)
    t_upper = np.percentile(boot_t_stats, 97.5)
    
    print(f"\nBootstrap t-dist: [{t_lower:.3f}, {t_upper:.3f}]")
    
    # Construct CI for parameter: [β - t_97.5 * SE, β - t_2.5 * SE]
    # Note: t_lower corresponds to left tail, so subtracted from estimate creates upper bound conceptually?
    # Standard formula: [β - t*(1-α/2)×SE, β - t*(α/2)×SE]
    se_orig = res.std_errors['Inter_Nbuy_Saiz']
    ci_lower_param = original_beta - t_upper * se_orig
    ci_upper_param = original_beta - t_lower * se_orig
    
    print(f"Original Estimate: {original_beta:.4f}")
    print(f"Wild Bootstrap 95% CI: [{ci_lower_param:.4f}, {ci_upper_param:.4f}]")
    
    # P-value (two-sided)
    # Proportion of |t*| > |t_orig|
    # But wait, t* is centered at 0 (approx). 
    # If we want to test H0: β=0, we should have imposed null.
    # Since we didn't impose null, we are testing if β_orig is significant using the bootstrap distribution of (β* - β_hat).
    # The CI check is sufficient. if 0 is outside CI.
    
    if ci_lower_param > 0 or ci_upper_param < 0:
        print("\n✓ SIGNIFICANT (0 outside CI)")
    else:
        print("\n~ NOT SIGNIFICANT (0 inside CI)")

if __name__ == "__main__":
    # Use smaller B for speed in agent loop, 199 is enough for rough check
    run_wild_bootstrap(B=199)
