# Detailed Replication Guide

This document provides a comprehensive guide for replicating the results in "The Friction Gate: Narrative Transmission in Housing Markets."

---

## Table of Contents

1. [Computational Environment](#computational-environment)
2. [Data Pipeline](#data-pipeline)
3. [Analysis Workflow](#analysis-workflow)
4. [Code-to-Paper Mapping](#code-to-paper-mapping)
5. [Expected Outputs](#expected-outputs)
6. [Troubleshooting](#troubleshooting)

---

## Computational Environment

### Hardware Requirements

- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~500MB free space
- **Internet**: Required only if refreshing Google Trends data

### Software Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.9+ | Tested on 3.9, 3.10, 3.11 |
| Pandas | 1.5+ | Data manipulation |
| NumPy | 1.21+ | Numerical computing |
| Matplotlib | 3.5+ | Visualization |
| LinearModels | 4.27+ | Panel regression |

### Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, matplotlib, linearmodels; print('All packages installed')"
```

---

## Data Pipeline

### Overview

The data pipeline constructs a quarterly panel of U.S. housing markets from three primary sources:

1. **Redfin Metro Market Tracker** - Housing transaction volume, prices, inventory
2. **FRED** - Macroeconomic controls (mortgage rates, unemployment)
3. **Google Trends** - Narrative attention indices at DMA level

### Pipeline Steps

#### Step 1: Crosswalk Construction (Optional)

**Script**: `code/finalize_crosswalk.py`

This creates the deterministic mapping between 201 Redfin metros and 127 Nielsen DMAs.

```bash
python code/finalize_crosswalk.py
```

**Output**: `data/mappings/metro_dma_crosswalk_deterministic.csv`

> **Note**: This step is optional. The crosswalk is already included in the repository.

#### Step 2: Main Data Pipeline

**Script**: `code/03_real_data_pipeline.py`

This is the primary data construction script. It:

1. Loads Redfin housing data (~108MB compressed)
2. Fetches/caches FRED macro variables
3. Fetches/caches Google Trends narrative indices
4. Constructs the quarterly panel
5. Computes standardized narrative indices
6. Generates lagged variables and controls

```bash
# Using cached data (recommended for replication)
python code/03_real_data_pipeline.py

# Force refresh of Trends data (requires internet, ~30 min)
# Edit script to set USE_CACHED_TRENDS = False
```

**Input Files**:
- `data/raw/redfin_metro.tsv.gz` (included)
- `data/raw/fred_cache/*.csv` (included)
- `data/raw/trends_cache/dma_trends_quarterly.csv` (included)

**Output**: `data/processed/panel_data_real.csv`

**Expected Console Output**:
```
Loading Redfin data...
Processing quarterly panel...
Fetching FRED data (cached)...
Loading Google Trends (cached)...
Final sample: 6984 observations, 201 metros, 127 DMAs
Panel saved to: data/processed/panel_data_real.csv
```

### Data Dictionary

| Variable | Description | Source |
|----------|-------------|--------|
| `region` | Metropolitan area name | Redfin |
| `dma_code` | Nielsen DMA identifier | Crosswalk |
| `quarter` | Quarter (YYYY-MM-DD) | Constructed |
| `volume_growth` | Δln(transaction volume) | Redfin |
| `n_buy` | Buy narrative index (z-standardized) | Google Trends |
| `n_risk` | Risk narrative index (z-standardized) | Google Trends |
| `mortgage_rate` | 30-year fixed mortgage rate | FRED |
| `unemployment` | Unemployment rate | FRED |
| `jumbo_exposure` | High-value transaction share | Redfin |

---

## Analysis Workflow

### Quick Run (Recommended)

```bash
python code/run_all.py
```

This master script executes all analyses in the correct order and generates all outputs.

### Manual Execution Order

If running scripts individually, execute in this order:

#### 1. Baseline Analysis
**Script**: `code/04_analysis_real.py`

Runs the main OLS and panel regressions for Table 2 (Baseline Results).

```bash
python code/04_analysis_real.py
```

**Key Outputs**:
- Console: Coefficient estimates, R², standard errors
- File: `output/regression_results_real.txt`

**Expected Results** (Phase 1.8B Deterministic Crosswalk):
```
BASELINE RESULTS
================
n_buy coefficient: -0.0326 (SE: 0.017, p=0.055)
n_risk coefficient: -0.0061
Valuation-Volume Ratio: 10.28
```

#### 2. Mechanism: Narrative × Friction Interactions
**Script**: `code/05_interaction_models.py`

Tests the "Friction Gate" hypothesis: narrative effects conditional on inventory constraints.

```bash
python code/05_interaction_models.py
```

**Key Outputs**:
- Interaction coefficients (n_buy × low_inventory)
- Subsample analysis by inventory terciles

#### 3. Mechanism: Supply Elasticity (Saiz)
**Script**: `code/07_mechanism_saiz.py`

Tests narrative effects by housing supply elasticity (Saiz, 2010).

```bash
python code/07_mechanism_saiz.py
```

**Key Outputs**:
- Analysis by elasticity quartiles
- High-friction vs. low-friction subsamples

#### 4. Robustness Checks
**Scripts**:
- `code/06_discriminant_validity.py` - Placebo tests
- `code/07_robustness.py` - Alternative specifications
- `code/11_wild_bootstrap.py` - Wild bootstrap inference

```bash
python code/06_discriminant_validity.py
python code/07_robustness.py
python code/11_wild_bootstrap.py
```

#### 5. Generate Tables
**Script**: `code/12_final_tables.py`

Creates LaTeX-formatted regression tables for the paper.

```bash
python code/12_final_tables.py
```

**Output**: `output/tables/` directory with `.tex` files

---

## Code-to-Paper Mapping

### Main Results

| Paper Element | Script | Output File | Line/Page |
|--------------|--------|-------------|-----------|
| **Table 2**: Baseline Panel Regression | `04_analysis_real.py` | `output/regression_results_real.txt` | Table 2 |
| **Table 3**: Mechanism - Inventory | `05_interaction_models.py` | Console output | Table 3 |
| **Table 4**: Mechanism - Supply Elasticity | `07_mechanism_saiz.py` | Console output | Table 4 |
| **Figure 2**: Narrative Time Series | `04_analysis_real.py` | `output/figures/narrative_series.pdf` | Figure 2 |
| **Figure 3**: Interaction Effects | `05_interaction_models.py` | `output/figures/interaction_plot.pdf` | Figure 3 |

### Robustness and Appendices

| Paper Element | Script | Notes |
|--------------|--------|-------|
| Appendix A: Data Construction | `03_real_data_pipeline.py` | Full pipeline documentation |
| Appendix B: Discriminant Validity | `06_discriminant_validity.py` | Placebo tests |
| Appendix C: Alternative Specifications | `07_robustness.py` | Robustness checks |
| Appendix D: Wild Bootstrap | `11_wild_bootstrap.py` | Inference with clustered errors |

### Sample Statistics

All sample statistics are defined in `code/constants.py` and should match:

```python
SAMPLE_STATS = {
    'n_observations': 6984,
    'n_metros': 201,
    'n_dmas': 127,
    'period_start': '2012Q1',
    'period_end': '2024Q4',
    'n_quarters': 52,
}
```

---

## Expected Outputs

### Directory Structure After Replication

```
output/
├── figures/
│   ├── narrative_series.pdf
│   ├── interaction_plot.pdf
│   ├── mechanism_inventory.pdf
│   └── mechanism_saiz.pdf
├── tables/
│   ├── table2_baseline.tex
│   ├── table3_interactions.tex
│   └── table4_mechanisms.tex
└── regression_results_real.txt
```

### Verification Checklist

After running the replication, verify:

- [ ] `panel_data_real.csv` exists in `data/processed/`
- [ ] Console shows: "Final sample: 6984 observations"
- [ ] Table 2 baseline coefficient: n_buy ≈ -0.033 (SE ≈ 0.017)
- [ ] VP Ratio reported as ~10.3
- [ ] All figures generated in `output/figures/`

---

## Troubleshooting

### Issue: Missing Data Files

**Symptom**: `FileNotFoundError: panel_data_real.csv not found`

**Solution**:
```bash
# Run data pipeline first
python code/03_real_data_pipeline.py
```

### Issue: Google Trends Rate Limiting

**Symptom**: `TooManyRequestsError` from pytrends

**Solution**: The repository includes cached Trends data. If refreshing:
```python
# In 03_real_data_pipeline.py, add delays between requests
import time
time.sleep(5)  # Add between API calls
```

### Issue: Memory Error on Data Load

**Symptom**: `MemoryError` when loading Redfin data

**Solution**:
```python
# Use chunking in 03_real_data_pipeline.py
df = pd.read_csv('redfin_metro.tsv.gz', compression='gzip', chunksize=100000)
```

### Issue: LaTeX Compilation Errors

**Symptom**: `! Undefined control sequence` in tables

**Solution**: Ensure required LaTeX packages are installed:
```bash
# In your LaTeX preamble
\usepackage{booktabs}
\usepackage{threeparttable}
\usepackage{siunitx}
```

### Issue: Different Results from Paper

**Checklist**:
1. [ ] Using deterministic crosswalk? (`metro_dma_crosswalk_deterministic.csv`)
2. [ ] Complete case analysis (no imputation)?
3. [ ] DMA-level clustering? (not metro-level)
4. [ ] Correct time period (2012Q1-2024Q4)?
5. [ ] Variables standardized within keyword?

If issues persist, check `code/constants.py` for verified statistics.

---

## Contact

For replication issues not resolved by this guide, please:

1. Check existing [GitHub Issues](https://github.com/qingsongcui/housing-narratives/issues)
2. Open a new issue with:
   - Python version (`python --version`)
   - Error message (full traceback)
   - Operating system
   - Steps to reproduce

---

*Last updated: February 2025*
