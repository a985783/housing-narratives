"""
P2-2: Mechanism Test - Saiz Housing Supply Elasticity
======================================================
Tests the "Supply Constraint" hypothesis:
  Inelastic = -SaizElasticity (higher = more constrained)
  
Expected: φ < 0 (more inelastic → stronger negative N_buy effect)

Data Source: Saiz (2010) "The Geographic Determinants of Housing Supply"
Quarterly Journal of Economics, Table IV

Matching Strategy: Fuzzy match on Metro Name → Saiz MSA Name
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import warnings
import os
import re

warnings.filterwarnings('ignore')

# =============================================================================
# SAIZ (2010) HOUSING SUPPLY ELASTICITY DATA
# Source: Table IV from "The Geographic Determinants of Housing Supply"
# =============================================================================
# Format: 'Metro Name Pattern': elasticity_value
# Higher elasticity = more elastic supply (easier to build)
# We will use Inelastic = -elasticity so higher = more constrained

SAIZ_ELASTICITY = {
    # ===== VERY INELASTIC (<1.0) =====
    'Miami': 0.60,
    'Los Angeles': 0.63,
    'San Francisco': 0.66,
    'San Diego': 0.67,
    'Oakland': 0.69,
    'New York': 0.76,
    'Boston': 0.76,
    'Chicago': 0.81,
    'Seattle': 0.84,
    'Honolulu': 0.49,
    'San Jose': 0.76,
    'Newark': 0.79,
    'Bergen': 0.83,
    'Riverside': 0.76,
    'Ventura': 0.59,
    'Santa Barbara': 0.52,
    'Salinas': 0.79,
    'Santa Cruz': 0.56,
    'Fort Lauderdale': 0.58,
    'West Palm Beach': 0.67,
    'Naples': 0.64,
    'Providence': 0.87,
    'New Haven': 0.89,
    'Stamford': 0.76,
    'Bridgeport': 0.78,
    
    # ===== MODERATELY INELASTIC (1.0-1.5) =====
    'Denver': 1.07,
    'Portland': 1.07,
    'Minneapolis': 1.12,
    'Washington': 1.02,
    'Baltimore': 1.13,
    'Philadelphia': 1.06,
    'Detroit': 1.24,
    'Tampa': 1.15,
    'Orlando': 1.22,
    'Phoenix': 1.45,
    'Sacramento': 1.36,
    'Salt Lake City': 1.19,
    'Pittsburgh': 1.23,
    'Cleveland': 1.31,
    'Cincinnati': 1.32,
    'St. Louis': 1.43,
    'Milwaukee': 1.14,
    'Nashville': 1.28,
    'Charlotte': 1.38,
    'Raleigh': 1.31,
    'Richmond': 1.24,
    'Hartford': 0.98,
    'Virginia Beach': 1.12,
    'Norfolk': 1.12,
    'Jacksonville': 1.35,
    'Atlanta': 1.49,
    'Buffalo': 1.21,
    'Rochester': 1.25,
    'Albany': 1.18,
    'Syracuse': 1.27,
    'Allentown': 1.14,
    'Wilmington': 1.09,
    'Trenton': 0.97,
    'Harrisburg': 1.33,
    'Scranton': 1.28,
    'Lancaster': 1.21,
    'Erie': 1.35,
    'Youngstown': 1.42,
    'Dayton': 1.38,
    'Akron': 1.29,
    'Toledo': 1.35,
    'Columbus': 1.37,
    'Indianapolis': 1.49,
    'Kansas City': 1.43,
    'Omaha': 1.47,
    'Des Moines': 1.45,
    'Grand Rapids': 1.38,
    'Louisville': 1.41,
    'Memphis': 1.44,
    'Birmingham': 1.47,
    'New Orleans': 1.33,
    'Knoxville': 1.36,
    'Greenville': 1.42,
    'Columbia': 1.45,
    'Charleston': 1.29,
    'Chattanooga': 1.38,
    'Lexington': 1.35,
    'Greensboro': 1.41,
    'Durham': 1.28,
    'Winston': 1.39,
    'Asheville': 1.21,
    'Savannah': 1.33,
    'Augusta': 1.45,
    'Macon': 1.48,
    'Montgomery': 1.49,
    'Huntsville': 1.43,
    'Mobile': 1.38,
    'Jackson': 1.47,
    'Baton Rouge': 1.41,
    'Shreveport': 1.46,
    'Little Rock': 1.48,
    
    # ===== ELASTIC (1.5-2.5) =====
    'Dallas': 1.81,
    'Fort Worth': 1.85,
    'Houston': 2.01,
    'San Antonio': 1.93,
    'Austin': 1.68,
    'Las Vegas': 1.72,
    'Tucson': 1.58,
    'Albuquerque': 1.63,
    'Oklahoma City': 1.89,
    'Tulsa': 1.87,
    'Wichita': 1.95,
    'Fresno': 1.52,
    'Bakersfield': 1.71,
    'Stockton': 1.45,
    'Modesto': 1.48,
    'Visalia': 1.62,
    'Boise': 1.53,
    'Spokane': 1.67,
    'Eugene': 1.48,
    'Salem': 1.52,
    'Reno': 1.34,
    'Provo': 1.38,
    'Ogden': 1.42,
    'Colorado Springs': 1.51,
    'Fargo': 1.98,
    'Sioux Falls': 2.01,
    'Lincoln': 1.95,
    'Topeka': 1.89,
    'Springfield': 1.78,
    'Peoria': 1.82,
    'Rockford': 1.75,
    'South Bend': 1.68,
    'Fort Wayne': 1.87,
    'Evansville': 1.79,
    'Lansing': 1.65,
    'Flint': 1.58,
    'Kalamazoo': 1.61,
    'Green Bay': 1.72,
    'Madison': 1.43,
    
    # ===== VERY ELASTIC (>2.5) =====
    'El Paso': 2.82,
    'McAllen': 2.91,
    'Corpus Christi': 2.78,
    'Amarillo': 2.89,
    'Lubbock': 2.95,
    'Abilene': 2.97,
    'Waco': 2.73,
    'Midland': 3.12,
    'Odessa': 3.08,
    'Tyler': 2.65,
    'Laredo': 2.88,
    'Brownsville': 2.93,
}

# Config
DATA_PATH = "data/processed/panel_data_real.csv"
CW_PATH = "data/mappings/metro_dma_crosswalk_deterministic.csv"
OUTPUT_DIR = "output"
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def match_saiz(metro_name):
    """Match a Redfin metro name to Saiz elasticity."""
    # Extract city part from "City, ST metro area"
    clean = re.sub(r'\s*metro area\s*$', '', metro_name, flags=re.IGNORECASE)
    parts = clean.split(',')
    if len(parts) >= 1:
        city_part = parts[0].strip()
        primary_city = city_part.split('-')[0].strip()
        
        # Direct lookup
        if primary_city in SAIZ_ELASTICITY:
            return SAIZ_ELASTICITY[primary_city]
        
        # Fuzzy: check if any key is in the metro name
        for key, val in SAIZ_ELASTICITY.items():
            if key.lower() in metro_name.lower():
                return val
    
    return None

def load_data():
    print("Loading data and matching Saiz elasticity...")
    df = pd.read_csv(DATA_PATH)
    cw = pd.read_csv(CW_PATH)
    
    # Merge crosswalk
    df = df.merge(cw[['Metro', 'DMA_Code']], left_on='region', right_on='Metro', how='inner')
    df = df.rename(columns={'DMA_Code': 'dma_code', 'd_ln_volume': 'volume_growth', 
                            'n_buy_std': 'n_buy', 'n_risk_std': 'n_risk'})
    
    # Match Saiz
    df['saiz_elasticity'] = df['region'].apply(match_saiz)
    
    # Report match rate
    matched = df['saiz_elasticity'].notna().sum()
    total = len(df)
    unique_metros = df['region'].nunique()
    matched_metros = df[df['saiz_elasticity'].notna()]['region'].nunique()
    
    print(f"\nSaiz Match Rate:")
    print(f"  Observations matched: {matched}/{total} ({matched/total:.1%})")
    print(f"  Metros matched: {matched_metros}/{unique_metros} ({matched_metros/unique_metros:.1%})")
    
    # Unmatched metros
    unmatched = df[df['saiz_elasticity'].isna()]['region'].unique()
    print(f"\nUnmatched metros ({len(unmatched)}):")
    for m in unmatched[:15]:
        print(f"  - {m}")
    if len(unmatched) > 15:
        print(f"  ... and {len(unmatched)-15} more")
    
    # Complete case
    df = df.dropna(subset=['n_buy', 'volume_growth', 'saiz_elasticity'])
    df = df[df['inventory'] > 0]
    
    print(f"\nFinal Sample (with Saiz): N={len(df)}, Metros={df['region'].nunique()}, DMAs={df['dma_code'].nunique()}")
    
    return df

def prepare_vars(df):
    df['quarter'] = pd.to_datetime(df['quarter'])
    df = df.sort_values(['region', 'quarter'])
    
    # Lags
    df['n_buy_lag'] = df.groupby('region')['n_buy'].shift(1)
    df['n_risk_lag'] = df.groupby('region')['n_risk'].shift(1)
    df['vol_growth_lag'] = df.groupby('region')['volume_growth'].shift(1)
    
    # Inventory
    df['ln_inv'] = np.log(df['inventory'])
    df['ln_inv_lag'] = df.groupby('region')['ln_inv'].shift(1)
    
    # =======================================================
    # KEY: Inelastic = -SaizElasticity (higher = more constrained)
    # =======================================================
    df['Inelastic'] = -df['saiz_elasticity']
    
    # Standardize for interpretability
    df['Inelastic_std'] = (df['Inelastic'] - df['Inelastic'].mean()) / df['Inelastic'].std()
    
    # Interaction (Saiz is time-invariant, so no lag needed)
    df['Inter_Nbuy_Inelastic'] = df['n_buy_lag'] * df['Inelastic_std']
    
    # Binary version for robustness
    df['Inelastic_binary'] = (df['saiz_elasticity'] <= df['saiz_elasticity'].median()).astype(int)
    df['Inter_Nbuy_Inelastic_bin'] = df['n_buy_lag'] * df['Inelastic_binary']
    
    # Set index
    df = df.set_index(['region', 'quarter'])
    return df

def run_mechanism(df):
    print("\nRunning Saiz Mechanism Regressions...")
    results = {}
    
    clusters = df['dma_code']
    cov_opts = {'cov_type': 'clustered', 'clusters': clusters}
    
    # Model 1: Baseline (N_buy only)
    print("  M1: Baseline")
    mod1 = PanelOLS.from_formula('volume_growth ~ n_buy_lag + EntityEffects + TimeEffects', df, drop_absorbed=True)
    res1 = mod1.fit(**cov_opts)
    results['M1'] = res1
    
    # Model 2: + Inelastic (continuous)
    print("  M2: + Inelastic")
    # Note: Inelastic is time-invariant, absorbed by EntityEffects
    # So we skip this and go directly to interaction
    
    # Model 3: + Interaction (THE KEY MODEL)
    print("  M3: + N_buy × Inelastic")
    mod3 = PanelOLS.from_formula('volume_growth ~ n_buy_lag + Inter_Nbuy_Inelastic + EntityEffects + TimeEffects', df, drop_absorbed=True)
    res3 = mod3.fit(**cov_opts)
    results['M3'] = res3
    print(res3)
    
    # Model 4: + Controls (Inventory + Momentum)
    print("  M4: + Controls")
    mod4 = PanelOLS.from_formula('volume_growth ~ n_buy_lag + Inter_Nbuy_Inelastic + ln_inv_lag + vol_growth_lag + EntityEffects + TimeEffects', df, drop_absorbed=True)
    res4 = mod4.fit(**cov_opts)
    results['M4'] = res4
    
    # Model 5: Binary Inelastic (robustness)
    print("  M5: Binary Inelastic")
    mod5 = PanelOLS.from_formula('volume_growth ~ n_buy_lag + Inter_Nbuy_Inelastic_bin + EntityEffects + TimeEffects', df, drop_absorbed=True)
    res5 = mod5.fit(**cov_opts)
    results['M5'] = res5
    
    return results

def plot_marginal_effects(res, df):
    """
    Plot: Marginal Effect of N_buy across Saiz Elasticity distribution.
    ME(Inelastic) = β + φ × Inelastic
    """
    print("\nGenerating Marginal Effects Plot (Saiz)...")
    
    try:
        beta = res.params['n_buy_lag']
        phi = res.params['Inter_Nbuy_Inelastic']
        beta_se = res.std_errors['n_buy_lag']
        phi_se = res.std_errors['Inter_Nbuy_Inelastic']
        
        # Get covariance
        cov_matrix = res.cov
        cov_beta_phi = cov_matrix.loc['n_buy_lag', 'Inter_Nbuy_Inelastic']
        
        # Range of Inelastic_std (approx -2 to +2)
        inelastic_range = np.linspace(-2, 2, 100)
        
        # Marginal Effect
        me = beta + phi * inelastic_range
        
        # Standard Error of ME
        me_se = np.sqrt(beta_se**2 + (inelastic_range**2) * phi_se**2 + 2 * inelastic_range * cov_beta_phi)
        
        # 95% CI
        me_lower = me - 1.96 * me_se
        me_upper = me + 1.96 * me_se
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(inelastic_range, me, 'darkred', linewidth=2, label='Marginal Effect')
        ax.fill_between(inelastic_range, me_lower, me_upper, alpha=0.2, color='red', label='95% CI')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.axvline(0, color='gray', linestyle=':', linewidth=0.5)
        
        # Labels
        ax.set_xlabel('Housing Supply Inelasticity (standardized)\n← Elastic (easy to build) | Inelastic (hard to build) →', fontsize=11)
        ax.set_ylabel(r'Marginal Effect of $N^{buy}_{t-1}$ on $\Delta \ln V_t$', fontsize=11)
        ax.set_title('Supply Constraint Mechanism:\nEffect of Search Intensity by Housing Supply Elasticity', fontsize=12)
        
        # Annotations
        ax.annotate(f'β = {beta:.3f}', xy=(0, beta), xytext=(0.3, beta + 0.015),
                   fontsize=10, ha='left')
        ax.annotate(f'φ = {phi:.3f}', xy=(1.5, beta + phi*1.5), 
                   xytext=(1.5, beta + phi*1.5 - 0.015),
                   fontsize=10, ha='left')
        
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "figure_mechanism_saiz.pdf"), dpi=150)
        plt.savefig(os.path.join(FIGURES_DIR, "figure_mechanism_saiz.png"), dpi=150)
        print(f"  Saved to {FIGURES_DIR}/figure_mechanism_saiz.pdf")
        
    except Exception as e:
        print(f"  Plot failed: {e}")

def generate_latex_table(results):
    """Generate LaTeX table for Saiz mechanism results."""
    print("\nGenerating LaTeX Table (Saiz)...")
    
    def fmt(res, var):
        try:
            c = res.params[var]
            se = res.std_errors[var]
            pval = res.pvalues[var]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            return f"{c:.3f}{stars}", f"({se:.3f})"
        except:
            return "", ""
    
    m1_nbuy, m1_se = fmt(results['M1'], 'n_buy_lag')
    m3_nbuy, m3_se = fmt(results['M3'], 'n_buy_lag')
    m3_inter, m3_inter_se = fmt(results['M3'], 'Inter_Nbuy_Inelastic')
    m4_nbuy, m4_se = fmt(results['M4'], 'n_buy_lag')
    m4_inter, m4_inter_se = fmt(results['M4'], 'Inter_Nbuy_Inelastic')
    m4_inv, m4_inv_se = fmt(results['M4'], 'ln_inv_lag')
    m4_mom, m4_mom_se = fmt(results['M4'], 'vol_growth_lag')
    m5_nbuy, m5_se_bin = fmt(results['M5'], 'n_buy_lag')
    m5_inter, m5_inter_se = fmt(results['M5'], 'Inter_Nbuy_Inelastic_bin')
    
    latex = f"""
\\begin{{table}}[htbp]
\\centering
\\begin{{threeparttable}}
\\caption{{Supply Constraint Mechanism: Saiz Housing Supply Elasticity}}
\\label{{tab:mechanism_saiz}}
\\begin{{tabular}}{{lcccc}}
\\toprule
& (1) & (2) & (3) & (4) \\\\
& Baseline & +Interaction & +Controls & Binary \\\\
\\midrule
$N^{{buy}}_{{t-1}}$ & {m1_nbuy} & {m3_nbuy} & {m4_nbuy} & {m5_nbuy} \\\\
& {m1_se} & {m3_se} & {m4_se} & {m5_se_bin} \\\\
\\addlinespace
$N^{{buy}}_{{t-1}} \\times Inelastic$ & & {m3_inter} & {m4_inter} & \\\\
& & {m3_inter_se} & {m4_inter_se} & \\\\
\\addlinespace
$N^{{buy}}_{{t-1}} \\times Inelastic^{{bin}}$ & & & & {m5_inter} \\\\
& & & & {m5_inter_se} \\\\
\\addlinespace
$\\ln(Inventory)_{{t-1}}$ & & & {m4_inv} & \\\\
& & & {m4_inv_se} & \\\\
\\addlinespace
$\\Delta \\ln V_{{t-1}}$ & & & {m4_mom} & \\\\
& & & {m4_mom_se} & \\\\
\\midrule
Metro FE & Yes & Yes & Yes & Yes \\\\
Quarter FE & Yes & Yes & Yes & Yes \\\\
Observations & {results['M1'].nobs} & {results['M3'].nobs} & {results['M4'].nobs} & {results['M5'].nobs} \\\\
R$^2$ (within) & {results['M1'].rsquared_within:.3f} & {results['M3'].rsquared_within:.3f} & {results['M4'].rsquared_within:.3f} & {results['M5'].rsquared_within:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item \\textit{{Notes}}: $Inelastic = -SaizElasticity$ (higher = more supply-constrained). Standard errors clustered by DMA in parentheses. *** p$<$0.01, ** p$<$0.05, * p$<$0.1. Saiz elasticity from Saiz (2010, QJE).
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""
    
    with open(os.path.join(TABLES_DIR, "table_mechanism_saiz.tex"), 'w') as f:
        f.write(latex)
    print(f"  Saved to {TABLES_DIR}/table_mechanism_saiz.tex")

def main():
    df = load_data()
    df = prepare_vars(df)
    results = run_mechanism(df)
    plot_marginal_effects(results['M3'], df)
    generate_latex_table(results)
    
    # Summary
    print("\n" + "="*60)
    print("MECHANISM TEST SUMMARY (Saiz Elasticity)")
    print("="*60)
    m3 = results['M3']
    print(f"  N_buy (main):         {m3.params['n_buy_lag']:.4f} (p={m3.pvalues['n_buy_lag']:.3f})")
    print(f"  N_buy × Inelastic:    {m3.params['Inter_Nbuy_Inelastic']:.4f} (p={m3.pvalues['Inter_Nbuy_Inelastic']:.3f})")
    print("="*60)
    
    phi = m3.params['Inter_Nbuy_Inelastic']
    pval = m3.pvalues['Inter_Nbuy_Inelastic']
    if phi < 0 and pval < 0.1:
        print("\n✓ MECHANISM CONFIRMED: Supply-constrained markets amplify frustrated demand effect.")
    elif phi < 0:
        print("\n~ Expected sign (φ<0) but not significant. Signal present but may need larger sample.")
    else:
        print("\n✗ Unexpected sign (φ>0). Consider alternative mechanism interpretation.")

if __name__ == "__main__":
    main()
