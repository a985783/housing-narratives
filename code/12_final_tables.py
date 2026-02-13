"""
Phase 3: Final Tables Generation
================================
1. Descriptive Statistics (Correct N=6,984)
2. Saiz Match Bias Table (Formatted for Appendix)
"""

import pandas as pd
import numpy as np
import os
from importlib.machinery import SourceFileLoader

# Load Saiz matching logic
saiz_module = SourceFileLoader("saiz", "code/07_mechanism_saiz.py").load_module()
match_saiz = saiz_module.match_saiz

DATA_PATH = "data/processed/panel_data_real.csv"
CW_PATH = "data/mappings/metro_dma_crosswalk_deterministic.csv"
OUTPUT_DIR = "output/tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_final_sample():
    # Load and replicate exact sample construction
    df = pd.read_csv(DATA_PATH)
    cw = pd.read_csv(CW_PATH)
    
    # Merge crosswalk
    df = df.merge(cw[['Metro', 'DMA_Code']], left_on='region', right_on='Metro', how='inner')
    
    # Drop missing essential variables
    df = df.dropna(subset=['n_buy_std', 'd_ln_volume', 'inventory'])
    
    # Filter non-zero inventory (log valid)
    df = df[df['inventory'] > 0]
    
    return df

def generate_descriptive_stats():
    print("Generating Table 1: Descriptive Statistics (N=6,984)...")
    df = load_final_sample()
    
    # Variables to summarize
    cols = {
        'd_ln_volume': 'Volume Growth',
        'price_growth': 'Price Growth',
        'n_buy_std': 'Buy-Side Search (Std)',
        'n_risk_std': 'Risk Search (Std)',
        'inventory': 'Inventory (Levels)'
    }
    
    stats = df[list(cols.keys())].describe().T
    stats = stats[['count', 'mean', 'std', 'min', '50%', 'max']]
    stats.columns = ['N', 'Mean', 'SD', 'Min', 'Median', 'Max']
    stats.index = [cols[c] for c in stats.index]
    
    # Format LaTeX
    latex = r"""
\begin{table}[htbp]
\centering
\begin{threeparttable}
\caption{Descriptive Statistics (Complete-Case Sample)}
\label{tab:descriptive_stats}
\begin{tabular}{lcccccc}
\toprule
Variable & N & Mean & SD & Min & Median & Max \\
\midrule
"""
    for idx, row in stats.iterrows():
        latex += f"{idx} & {int(row['N']):,} & {row['Mean']:.3f} & {row['SD']:.3f} & {row['Min']:.3f} & {row['Median']:.3f} & {row['Max']:.3f} \\\\\n"
        
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes}: Sample consists of 143 U.S. metropolitan areas (99 DMAs) from 2012Q1 to 2024Q4. All variables are quarterly.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    with open(f"{OUTPUT_DIR}/table_descriptive_stats.tex", 'w') as f:
        f.write(latex)
    print("Saved table_descriptive_stats.tex")

def generate_match_bias_table():
    print("Generating Saiz Match Bias Table...")
    df = pd.read_csv(DATA_PATH)
    cw = pd.read_csv(CW_PATH)
    
    # Base sample (before dropna) but after crosswalk
    df = df.merge(cw[['Metro', 'DMA_Code']], left_on='region', right_on='Metro', how='inner')
    
    vars_map = {
        'homes_sold': 'Avg. Monthly Sales',
        'median_sale_price': 'Avg. Median Price (\\$)',
        'inventory': 'Avg. Inventory',
        'n_buy_std': 'Search Intensity',
        'd_ln_volume': 'Volume Growth'
    }
    
    # Get unique metros statistics
    metro_stats = df.groupby('region').agg({
        'homes_sold': 'mean',
        'median_sale_price': 'mean',
        'inventory': 'mean',
        'n_buy_std': 'mean',
        'd_ln_volume': 'mean'
    }).reset_index()
    
    metro_stats['Matched'] = metro_stats['region'].apply(lambda x: match_saiz(x) is not None)
    
    # Calculate means (numeric only)
    grp = metro_stats.groupby('Matched')[list(vars_map.keys())].mean()
    
    # T-tests
    from scipy.stats import ttest_ind
    
    latex = r"""
\begin{table}[htbp]
\centering
\begin{threeparttable}
\caption{Saiz Match Bias Diagnostic}
\label{tab:match_bias}
\begin{tabular}{lccc}
\toprule
Variable (Mean) & Matched (N=145) & Unmatched (N=57) & Difference \\
\midrule
"""
    
    for v, label in vars_map.items():
        m_matched = grp.loc[True, v]
        m_unmatched = grp.loc[False, v]
        diff = m_matched - m_unmatched
        
        # T-test
        s1 = metro_stats[metro_stats['Matched']==True][v].dropna()
        s2 = metro_stats[metro_stats['Matched']==False][v].dropna()
        t, p = ttest_ind(s1, s2, equal_var=False)
        stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else ""
        
        latex += f"{label} & {m_matched:,.2f} & {m_unmatched:,.2f} & {diff:+,.2f}{stars} \\\\\n"
        
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes}: Comparison of metro-level averages for metros matched to Saiz (2010) elasticity data vs. unmatched. Matched metros form the subsample for mechanism testing.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
    with open(f"{OUTPUT_DIR}/table_match_bias.tex", 'w') as f:
        f.write(latex)
    print("Saved table_match_bias.tex")

if __name__ == "__main__":
    generate_descriptive_stats()
    generate_match_bias_table()
