"""
ECR-CAS Housing Cycles - Sample Constants (Single Source of Truth)
===================================================================
This file defines all sample statistics to ensure consistency across
the paper, tables, and analysis scripts.

LAST VERIFIED: 2026-02-01 (Phase 1.8B Deterministic Crosswalk)
"""

# =============================================================================
# SAMPLE FUNNEL (For Data Section Table)
# =============================================================================
SAMPLE_FUNNEL = {
    'raw_metros': 300,           # Redfin top metros
    'mapped_to_dma': 202,        # After deterministic crosswalk
    'trends_available': 143,     # Metros with Google Trends data
    'complete_case': 143,        # Final sample (no missing narratives)
}

# =============================================================================
# FINAL SAMPLE STATISTICS
# =============================================================================
SAMPLE_STATS = {
    'n_observations': 6984,
    'n_metros': 201,
    'n_dmas': 127,
    'period_start': '2012Q1',
    'period_end': '2024Q4',
    'n_quarters': 52,
}

# =============================================================================
# KEY REGRESSION RESULTS (Phase 1.8B)
# =============================================================================
BASELINE_RESULTS = {
    'n_buy_coef': -0.0326,
    'n_buy_se': 0.017,
    'n_buy_pval': 0.055,
    'n_risk_coef': -0.0061,
    'vp_ratio': 10.28,
}

MECHANISM_RESULTS = {
    'n_buy_main': -0.0462,
    'n_buy_main_pval': 0.006,
}

# =============================================================================
# CROSSWALK ARTIFACT
# =============================================================================
CROSSWALK_FILE = 'data/mappings/metro_dma_crosswalk_deterministic.csv'
AUDIT_FILE = 'data/mappings/crosswalk_audit.csv'

# =============================================================================
# LaTeX MACROS (for inclusion in paper)
# =============================================================================
def generate_latex_macros():
    """Generate LaTeX macros for sample stats."""
    return f"""
% Sample Statistics (Auto-generated from constants.py)
\\newcommand{{\\SampleN}}{{{SAMPLE_STATS['n_observations']:,}}}
\\newcommand{{\\SampleMetros}}{{{SAMPLE_STATS['n_metros']}}}
\\newcommand{{\\SampleDMAs}}{{{SAMPLE_STATS['n_dmas']}}}
\\newcommand{{\\SamplePeriod}}{{{SAMPLE_STATS['period_start']}--{SAMPLE_STATS['period_end']}}}

% Key Coefficients
\\newcommand{{\\BetaBuy}}{{{BASELINE_RESULTS['n_buy_coef']:.3f}}}
\\newcommand{{\\VPRatio}}{{{BASELINE_RESULTS['vp_ratio']:.1f}}}
"""

if __name__ == "__main__":
    print("Sample Statistics:")
    for k, v in SAMPLE_STATS.items():
        print(f"  {k}: {v}")
    print("\nLaTeX Macros:")
    print(generate_latex_macros())
