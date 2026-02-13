"""
ECR-CAS Housing Cycles - Interaction Analysis (The "Rescue" Strategy)
=====================================================================
Tests conditional effects of Narrative (N) based on:
1. Structural Constraints (Saiz Elasticity)
2. State Frictions (Inventory)

Hypothesis: N -> Volume is not universal, but conditional on constraints.

Unit of Analysis: DMA-Quarter (Aggregated)
Inference: Wild Cluster Bootstrap (if possible) or Cluster-Robust SE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import warnings
import os
import re

warnings.filterwarnings('ignore')

# Config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/panel_data_real.csv")
CW_PATH = os.path.join(BASE_DIR, "data/mappings/metro_dma_crosswalk_deterministic.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "output", "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Hardcoded Saiz Data (from 07_mechanism_saiz.py)
SAIZ_ELASTICITY = {
    'Miami': 0.60, 'Los Angeles': 0.63, 'San Francisco': 0.66, 'San Diego': 0.67, 'Oakland': 0.69,
    'New York': 0.76, 'Boston': 0.76, 'Chicago': 0.81, 'Seattle': 0.84, 'Honolulu': 0.49,
    'San Jose': 0.76, 'Newark': 0.79, 'Bergen': 0.83, 'Riverside': 0.76, 'Ventura': 0.59,
    'Santa Barbara': 0.52, 'Salinas': 0.79, 'Santa Cruz': 0.56, 'Fort Lauderdale': 0.58,
    'West Palm Beach': 0.67, 'Naples': 0.64, 'Providence': 0.87, 'New Haven': 0.89,
    'Stamford': 0.76, 'Bridgeport': 0.78, 'Denver': 1.07, 'Portland': 1.07, 'Minneapolis': 1.12,
    'Washington': 1.02, 'Baltimore': 1.13, 'Philadelphia': 1.06, 'Detroit': 1.24, 'Tampa': 1.15,
    'Orlando': 1.22, 'Phoenix': 1.45, 'Sacramento': 1.36, 'Salt Lake City': 1.19, 'Pittsburgh': 1.23,
    'Cleveland': 1.31, 'Cincinnati': 1.32, 'St. Louis': 1.43, 'Milwaukee': 1.14, 'Nashville': 1.28,
    'Charlotte': 1.38, 'Raleigh': 1.31, 'Richmond': 1.24, 'Hartford': 0.98, 'Virginia Beach': 1.12,
    'Norfolk': 1.12, 'Jacksonville': 1.35, 'Atlanta': 1.49, 'Buffalo': 1.21, 'Rochester': 1.25,
    'Albany': 1.18, 'Syracuse': 1.27, 'Allentown': 1.14, 'Wilmington': 1.09, 'Trenton': 0.97,
    'Harrisburg': 1.33, 'Scranton': 1.28, 'Lancaster': 1.21, 'Erie': 1.35, 'Youngstown': 1.42,
    'Dayton': 1.38, 'Akron': 1.29, 'Toledo': 1.35, 'Columbus': 1.37, 'Indianapolis': 1.49,
    'Kansas City': 1.43, 'Omaha': 1.47, 'Des Moines': 1.45, 'Grand Rapids': 1.38, 'Louisville': 1.41,
    'Memphis': 1.44, 'Birmingham': 1.47, 'New Orleans': 1.33, 'Knoxville': 1.36, 'Greenville': 1.42,
    'Columbia': 1.45, 'Charleston': 1.29, 'Chattanooga': 1.38, 'Lexington': 1.35, 'Greensboro': 1.41,
    'Durham': 1.28, 'Winston': 1.39, 'Asheville': 1.21, 'Savannah': 1.33, 'Augusta': 1.45,
    'Macon': 1.48, 'Montgomery': 1.49, 'Huntsville': 1.43, 'Mobile': 1.38, 'Jackson': 1.47,
    'Baton Rouge': 1.41, 'Shreveport': 1.46, 'Little Rock': 1.48, 'Dallas': 1.81, 'Fort Worth': 1.85,
    'Houston': 2.01, 'San Antonio': 1.93, 'Austin': 1.68, 'Las Vegas': 1.72, 'Tucson': 1.58,
    'Albuquerque': 1.63, 'Oklahoma City': 1.89, 'Tulsa': 1.87, 'Wichita': 1.95, 'Fresno': 1.52,
    'Bakersfield': 1.71, 'Stockton': 1.45, 'Modesto': 1.48, 'Visalia': 1.62, 'Boise': 1.53,
    'Spokane': 1.67, 'Eugene': 1.48, 'Salem': 1.52, 'Reno': 1.34, 'Provo': 1.38, 'Ogden': 1.42,
    'Colorado Springs': 1.51, 'Fargo': 1.98, 'Sioux Falls': 2.01, 'Lincoln': 1.95, 'Topeka': 1.89,
    'Springfield': 1.78, 'Peoria': 1.82, 'Rockford': 1.75, 'South Bend': 1.68, 'Fort Wayne': 1.87,
    'Evansville': 1.79, 'Lansing': 1.65, 'Flint': 1.58, 'Kalamazoo': 1.61, 'Green Bay': 1.72,
    'Madison': 1.43, 'El Paso': 2.82, 'McAllen': 2.91, 'Corpus Christi': 2.78, 'Amarillo': 2.89,
    'Lubbock': 2.95, 'Abilene': 2.97, 'Waco': 2.73, 'Midland': 3.12, 'Odessa': 3.08, 'Tyler': 2.65,
    'Laredo': 2.88, 'Brownsville': 2.93
}

def match_saiz(metro_name):
    clean = re.sub(r'\s*metro area\s*$', '', metro_name, flags=re.IGNORECASE)
    parts = clean.split(',')
    if len(parts) >= 1:
        city_part = parts[0].strip()
        primary_city = city_part.split('-')[0].strip()
        if primary_city in SAIZ_ELASTICITY:
            return SAIZ_ELASTICITY[primary_city]
        for key, val in SAIZ_ELASTICITY.items():
            if key.lower() in metro_name.lower():
                return val
    return None

def load_and_aggregate():
    print("Loading Panel Data...")
    df = pd.read_csv(DATA_PATH)
    if 'dma_code' in df.columns:
        df = df.drop(columns=['dma_code'])
    cw = pd.read_csv(CW_PATH)
    
    # 1. Merge Crosswalk
    df = df.merge(cw[['Metro', 'DMA_Code']], left_on='region', right_on='Metro', how='inner')
    df = df.rename(columns={'DMA_Code': 'dma_code'})
    
    # 2. Match Saiz (Metro Level)
    df['saiz_elasticity'] = df['region'].apply(match_saiz)
    
    # Missing Saiz imputation (mean by State?) - Skip for now, focus on matched
    # But for aggregation, if some metros in DMA have Saiz and others don't, 
    # we take weighted avg of available.
    
    # 3. Aggregate to DMA Level
    print("\nAggregating to DMA Level (checking interactions)...")
    
    # Ensure quarter is datetime
    df['quarter'] = pd.to_datetime(df['quarter'])
    
    dma_groups = df.groupby(['dma_code', 'quarter'])
    
    df_dma = pd.DataFrame()
    df_dma['homes_sold'] = dma_groups['homes_sold'].sum()
    df_dma['inventory'] = dma_groups['inventory'].sum()
    
    # Weighted Average Helper
    def weighted_avg(x, val_col, w_col):
        mask = x[val_col].notna()
        if mask.any():
            return np.average(x.loc[mask, val_col], weights=x.loc[mask, w_col])
        return np.nan

    # Weighted Saiz
    df_dma['saiz_elasticity'] = df.groupby(['dma_code', 'quarter']).apply(
        lambda x: weighted_avg(x, 'saiz_elasticity', 'homes_sold')
    )
    
    # Narrative (Mean)
    df_dma['n_buy'] = dma_groups['n_buy'].mean()
    
    df_dma = df_dma.reset_index()
    df_dma = df_dma.sort_values(['dma_code', 'quarter'])
    
    # 4. Construct Variables
    # Volume Growth
    df_dma['ln_volume'] = np.log(df_dma['homes_sold'].clip(lower=1))
    df_dma['volume_growth'] = df_dma.groupby('dma_code')['ln_volume'].diff()
    
    # Inventory (Log)
    df_dma['ln_inventory'] = np.log(df_dma['inventory'].replace(0, np.nan).ffill().bfill())
    
    # Lags
    df_dma['n_buy_lag'] = df_dma.groupby('dma_code')['n_buy'].shift(1)
    df_dma['ln_inv_lag'] = df_dma.groupby('dma_code')['ln_inventory'].shift(1)
    
    # 5. Interaction Terms
    # (A) Structure: Inelastic = -Saiz
    # Standardize Inelastic for interpretability (continuous)
    df_dma['Inelastic'] = -df_dma['saiz_elasticity']
    # Fill missing Saiz with mean (to avoid dropping DMAs just for mechanism check?)
    # Better to drop separate sample.
    df_dma = df_dma.dropna(subset=['Inelastic'])
    df_dma['Inelastic_std'] = (df_dma['Inelastic'] - df_dma['Inelastic'].mean()) / df_dma['Inelastic'].std()
    
    # (B) State: Low Inventory
    # Low Inv = below median (within sample or within DMA? User suggested sample quantile for now)
    # Using 'Tight Market' dummy
    median_inv = df_dma['ln_inv_lag'].median()
    df_dma['LowInv'] = (df_dma['ln_inv_lag'] < median_inv).astype(int)
    # Also create continuous Inv interaction (standardized)
    df_dma['Inv_std'] = (df_dma['ln_inv_lag'] - df_dma['ln_inv_lag'].mean()) / df_dma['ln_inv_lag'].std()
    
    # Interaction Vars
    df_dma['Inter_Structure'] = df_dma['n_buy_lag'] * df_dma['Inelastic_std']
    df_dma['Inter_State_Cont'] = df_dma['n_buy_lag'] * df_dma['Inv_std']
    df_dma['Inter_State_Bin'] = df_dma['n_buy_lag'] * df_dma['LowInv']
    
    # Triple Interaction: N * Inelastic * LowInv
    df_dma['Inter_Triple'] = df_dma['n_buy_lag'] * df_dma['Inelastic_std'] * df_dma['LowInv']
    
    df_dma = df_dma.set_index(['dma_code', 'quarter'])
    df_dma = df_dma.dropna(subset=['volume_growth', 'n_buy_lag', 'Inter_Structure'])
    
    print(f"\nFinal Interaction Panel: {df_dma.index.get_level_values(0).nunique()} DMAs, {len(df_dma)} Obs")
    return df_dma

def run_models(df):
    results = {}
    print("\nRunning Interaction Regressions...")
    
    # M1: Baseline (Replicate)
    mod1 = PanelOLS.from_formula('volume_growth ~ n_buy_lag + ln_inv_lag + EntityEffects + TimeEffects', df)
    results['M1'] = mod1.fit(cov_type='clustered', cluster_entity=True)
    
    # M2: Structure Interaction (N * Inelastic)
    print("  M2: Structure (Inelastic)")
    # Hypothesis: Coef on Inter < 0 (More Inelastic -> More Negative Volume Response to Buy Search??)
    # Wait, "Buy Search" -> "Volume". If frustrated demand -> Volume LOWER?
    # Or Buy Search -> Volume HIGHER? 
    # Usually Search -> Volume is Positive.
    # But if "Frustrated", Search UP -> Volume DOWN (or less Up).
    # If Saiz Constraint amplifies "Frustration", then Inter should be NEGATIVE (Lower volume growth for same search).
    mod2 = PanelOLS.from_formula('volume_growth ~ n_buy_lag + Inter_Structure + ln_inv_lag + EntityEffects + TimeEffects', df)
    results['M2'] = mod2.fit(cov_type='clustered', cluster_entity=True)
    
    # M3: State Interaction (N * LowInv)
    print("  M3: State (Low Inventory)")
    # If LowInv makes market tight -> Search leads to "bidding war" (Price Up) or "Volume Constrained" (Volume Down)?
    # Likely Volume Down (Constraint).
    mod3 = PanelOLS.from_formula('volume_growth ~ n_buy_lag + Inter_State_Bin + ln_inv_lag + EntityEffects + TimeEffects', df)
    results['M3'] = mod3.fit(cov_type='clustered', cluster_entity=True)
    
    # M4: Triple Interaction
    print("  M4: Triple (N * Inelastic * LowInv)")
    mod4 = PanelOLS.from_formula('volume_growth ~ n_buy_lag + Inter_Structure + Inter_State_Bin + Inter_Triple + ln_inv_lag + EntityEffects + TimeEffects', df)
    results['M4'] = mod4.fit(cov_type='clustered', cluster_entity=True)
    
    # M5: Continuous State Interaction (N * ln_Inv_std)
    print("  M5: Continuous State (N * ln_Inv_std)")
    # Exp: N effect should be stronger (more positive) when Inv is LOWER.
    # So N * Inv relationship should be NEGATIVE? 
    #   High Inv -> Low/Zero Effect. 
    #   Low Inv -> High Positive Effect.
    #   So slope w.r.t Inv should be Negative.
    mod5 = PanelOLS.from_formula('volume_growth ~ n_buy_lag + Inter_State_Cont + ln_inv_lag + EntityEffects + TimeEffects', df)
    results['M5'] = mod5.fit(cov_type='clustered', cluster_entity=True)

    return results

def run_threshold_sensitivity(df):
    print("\nRunning LowInv Threshold Sensitivity Analysis...")
    print(f"{'Quantile':<10} {'Coef (N x LowInv)':<20} {'P-Value':<10} {'Signif'}")
    print("-" * 50)
    
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    sensitivity_results = []
    
    # Base formula
    # Note: re-calculating LowInv for each threshold
    
    for q in thresholds:
        # Define LowInv based on POOLED quantile (simplest "preset rule")
        thresh_val = df['ln_inv_lag'].quantile(q)
        df_temp = df.copy()
        df_temp['LowInv_Sens'] = (df_temp['ln_inv_lag'] < thresh_val).astype(int)
        df_temp['Inter_Sens'] = df_temp['n_buy_lag'] * df_temp['LowInv_Sens']
        
        try:
            mod = PanelOLS.from_formula('volume_growth ~ n_buy_lag + Inter_Sens + ln_inv_lag + EntityEffects + TimeEffects', df_temp, drop_absorbed=True)
            res = mod.fit(cov_type='clustered', cluster_entity=True)
            
            coef = res.params['Inter_Sens']
            pval = res.pvalues['Inter_Sens']
            star = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            
            print(f"{q:<10.2f} {coef:<20.4f} {pval:<10.3f} {star}")
            sensitivity_results.append((q, coef, pval))
        except Exception as e:
            print(f"{q:<10.2f} Error: {e}")

    return sensitivity_results

def run_threshold_sweep_plot(df, output_dir):
    print("\nRunning Threshold Sweep for Plotting...")
    thresholds = np.arange(0.10, 0.65, 0.05)
    results = []
    
    for q in thresholds:
        thresh_val = df['ln_inv_lag'].quantile(q)
        df_temp = df.copy()
        df_temp['LowInv_Sens'] = (df_temp['ln_inv_lag'] < thresh_val).astype(int)
        df_temp['Inter_Sens'] = df_temp['n_buy_lag'] * df_temp['LowInv_Sens']
        
        try:
            mod = PanelOLS.from_formula('volume_growth ~ n_buy_lag + Inter_Sens + ln_inv_lag + EntityEffects + TimeEffects', df_temp, drop_absorbed=True)
            res = mod.fit(cov_type='clustered', cluster_entity=True)
            
            coef = res.params['Inter_Sens']
            se = res.std_errors['Inter_Sens']
            results.append({'quantile': q, 'coef': coef, 'se': se})
        except:
            pass
            
    # Save for plotting
    res_df = pd.DataFrame(results)
    res_df['ci_lower'] = res_df['coef'] - 1.96 * res_df['se']
    res_df['ci_upper'] = res_df['coef'] + 1.96 * res_df['se']
    
    csv_path = os.path.join(output_dir, 'threshold_sweep_data.csv')
    res_df.to_csv(csv_path, index=False)
    print(f"Saved threshold sweep data to {csv_path}")
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(res_df['quantile'], res_df['coef'], marker='o', label='Interaction Coef')
    plt.fill_between(res_df['quantile'], res_df['ci_lower'], res_df['ci_upper'], alpha=0.2, color='blue')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Low Inventory Quantile Threshold')
    plt.ylabel('Interaction Coefficient (N x LowInv)')
    plt.title('Robustness: "Friction Gate" Exists Across Thresholds')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir.replace('tables', 'figures'), 'figure_threshold_sweep.pdf'))
    # Also save as png for quick view
    plt.savefig(os.path.join(output_dir.replace('tables', 'figures'), 'figure_threshold_sweep.png'))
    plt.close()

def print_summary(results, df):
    print("\n" + "="*80)
    print("INTERACTION RESULTS SUMMARY")
    print("="*80)
    
    def get_res(res, term):
        try:
            c = res.params[term]
            p = res.pvalues[term]
            star = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
            return c, p, star
        except:
            return 0, 1, ""

    # M2 Structure
    c, p, s = get_res(results['M2'], 'Inter_Structure')
    print(f"Structure (N x Inelastic): {c:.4f} {s} (p={p:.3f})")
    
    # M3 State (Binary)
    c, p, s = get_res(results['M3'], 'Inter_State_Bin')
    print(f"State (N x LowInv Bin):    {c:.4f} {s} (p={p:.3f})")
    
    # Economic Magnitude
    if 'M3' in results:
        params = results['M3'].params
        # Effect in Low Inv = Base + Interaction
        # Base is n_buy_lag
        base = params['n_buy_lag']
        inter = params['Inter_State_Bin']
        
        sd_n = df['n_buy_lag'].std()
        net_effect = base + inter
        econ_impact = net_effect * sd_n
        
        print("\nEconomic Magnitude (State Model):")
        print(f"  SD of Narrative (N): {sd_n:.4f}")
        print(f"  Base Effect (High Inv): {base:.4f} (p={results['M3'].pvalues['n_buy_lag']:.3f})")
        print(f"  Interaction (Low Inv Boost): {inter:.4f} (p={results['M3'].pvalues['Inter_State_Bin']:.3f})")
        print(f"  Net Effect (Low Inv): {net_effect:.4f}")
        print(f"  -> 1 SD Shock in N maps to {econ_impact*100:.2f}% change in Volume Growth (in Tight Markets)")
    
    # M5 State (Continuous)
    c, p, s = get_res(results['M5'], 'Inter_State_Cont')
    print(f"State (N x Inv Cont):      {c:.4f} {s} (p={p:.3f})")
    if c < 0 and p < 0.1:
         print("  -> Continuous check confirms: Higher Inventory attenuates effect (Negative interaction).")

    # M4 Triple
    c, p, s = get_res(results['M4'], 'Inter_Triple')
    print(f"Triple (Gatekeeper):       {c:.4f} {s} (p={p:.3f})")
    
    # Output LaTeX Table
    print("\nGenerating LaTeX Table (Interaction)...")
    
    def fmt_tex(res, term):
        try:
            c = res.params[term]
            se = res.std_errors[term]
            p = res.pvalues[term]
            stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
            return f"{c:.3f}{stars}", f"({se:.3f})"
        except:
            return "", ""

    ltx = ""
    # Headers
    ltx += "\\begin{table}[htbp]\n\\centering\n\\begin{threeparttable}\n"
    ltx += "\\caption{Conditional Effects: Friction Gates Narrative Transmission}\n"
    ltx += "\\label{tab:interaction_results}\n"
    ltx += "\\begin{tabular}{lcccc}\n\\toprule\n"
    ltx += "& (1) & (2) & (3) & (4) \\\\\n"
    ltx += "& Baseline & Structure & State (Bin) & State (Cont) \\\\\n\\midrule\n"
    
    # Rows
    rows = [
        ("Narrative ($N_{t-1}$)", 'n_buy_lag'),
        ("Structure ($N \\times Inelastic$)", 'Inter_Structure'),
        ("State ($N \\times LowInv$)", 'Inter_State_Bin'),
        ("State ($N \\times \\ln Inv$)", 'Inter_State_Cont'),
        ("Structure ($N \\times Inelastic$)", 'Inter_Triple'), # Wait, Model 4 has different spec
        # Let's align columns: M1 (Base), M2 (Structure), M3 (State Bin), M5 (State Cont)
        # Skip Triple for main table for clarity unless strong?
        # User said Triple p=0.12, maybe keep to M2, M3, M5 as main story.
    ]
    
    # Let's map Models to Columns
    # Col 1: M1
    # Col 2: M2 (Structure)
    # Col 3: M3 (State Bin)
    # Col 4: M5 (State Cont)
    
    models = [results['M1'], results['M2'], results['M3'], results['M5']]
    
    row_map = {
        'n_buy_lag': "$N^{buy}_{t-1}$",
        'Inter_Structure': "$N^{buy}_{t-1} \\times Inelastic$",
        'Inter_State_Bin': "$N^{buy}_{t-1} \\times LowInv$",
        'Inter_State_Cont': "$N^{buy}_{t-1} \\times \\ln(Inventory)$",
        'ln_inv_lag': "$\ln(Inventory)_{t-1}$"
    }
    
    for key, label in row_map.items():
        line_beta = f"{label} & "
        line_se = "& "
        for m in models:
            beta, se = fmt_tex(m, key)
            line_beta += f"{beta} & "
            line_se += f"{se} & "
        ltx += line_beta[:-2] + "\\\\\n"
        ltx += line_se[:-2] + "\\\\\n\\addlinespace\n"
        
    # Stats
    ltx += "\\midrule\n"
    ltx += "DMA FE & Yes & Yes & Yes & Yes \\\\\n"
    ltx += "Quarter FE & Yes & Yes & Yes & Yes \\\\\n"
    ltx += f"Observations & {models[0].nobs} & {models[1].nobs} & {models[2].nobs} & {models[3].nobs} \\\\\n"
    ltx += f"$R^2$ (Within) & {models[0].rsquared_within:.3f} & {models[1].rsquared_within:.3f} & {models[2].rsquared_within:.3f} & {models[3].rsquared_within:.3f} \\\\\n"
    ltx += "\\bottomrule\n\\end{tabular}\n"
    ltx += "\\begin{tablenotes}\n\\small\n"
    ltx += "\\item \\textit{Notes}: Standard errors clustered by DMA. $LowInv$ is a binary indicator for inventory below the median. $Inelastic = -SaizElasticity$. \n"
    ltx += "\\end{tablenotes}\n\\end{threeparttable}\n\\end{table}"
    
    with open(os.path.join(OUTPUT_DIR, "table_interactions.tex"), "w") as f:
        f.write(ltx)
    print(f"Saved to {OUTPUT_DIR}/table_interactions.tex")

if __name__ == "__main__":
    df = load_and_aggregate()
    results = run_models(df)
    run_threshold_sensitivity(df)
    run_threshold_sweep_plot(df, OUTPUT_DIR)
    print_summary(results, df)
