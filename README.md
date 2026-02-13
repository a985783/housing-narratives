# Replication Package: The Friction Gate
## Narrative Transmission in Housing Markets

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## 📋 Overview

This repository contains the complete replication package for the paper **"The Friction Gate: Narrative Transmission in Housing Markets"** submitted to the *Journal of Economic Behavior & Organization*.

### Research Summary

This paper addresses a fundamental puzzle in narrative economics: the transmission mechanism from public attention to aggregate economic activity. While narratives are increasingly recognized as drivers of economic cycles, empirical evidence on their predictive power remains mixed. We reconcile these conflicting findings by proposing and testing a **"Friction Gate" theory**.

Using a novel identification strategy that corrects for cross-market comparability in Google Trends data via pooled within-keyword z-standardization, we analyze a rigorously constructed panel of **127 U.S. Designated Market Areas (DMAs) from 2012–2024**. Our results demonstrate that narrative attention is conditionally powerful: it significantly predicts transaction volume only in high-friction regimes—specifically when inventory is low and supply is structurally inelastic.

### Key Contributions

1. **Rigorous Identification**: We establish a robust null baseline for the average effect of narrative attention, showing that previous findings of "universal" effects may be artifacts of measurement error.
2. **The Friction Gate Theory**: We provide evidence that market frictions are not merely impediments to efficiency but act as conductors that transmit narrative signals into the real economy.
3. **Complex Systems Evidence**: Our findings align with a complex adaptive systems (CAS) view, linking narrative economics with the physics of phase transitions and state-dependent feedback loops.

---

## 📊 Data Availability Statement

### Data Sources

This replication package includes:

| Data Source | Included | Description |
|------------|----------|-------------|
| **Redfin Metro Data** | ✅ Full | Housing market data for top 300 U.S. metros (2012-2024) |
| **FRED Macro Variables** | ✅ Cached | 30-year mortgage rates (MORTGAGE30US) and unemployment (UNRATE) |
| **Google Trends** | ✅ Cached | DMA-level narrative indices for buy and risk keywords |
| **Metro-DMA Crosswalk** | ✅ Full | Deterministic mapping of 201 metros to 127 DMAs |

### Sample Statistics

- **Observations**: 6,984
- **Metropolitan Areas**: 201 metros → 127 DMAs
- **Time Period**: 2012Q1 – 2024Q4 (52 quarters)
- **Data Files**: `data/processed/panel_data_real.csv`

---

## 💻 Software & Hardware Requirements

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 8GB (16GB recommended for full pipeline)
- **Storage**: ~500MB for data and outputs
- **Python**: 3.9 or higher

### Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `pandas` >= 1.5.0
- `numpy` >= 1.21.0
- `matplotlib` >= 3.5.0
- `linearmodels` >= 4.27
- `requests` >= 2.28.0
- `pytrends` >= 4.9.0 (optional, for data refresh)

### Estimated Runtime

- **Full pipeline**: ~45 minutes (including Google Trends refresh)
- **Analysis only**: ~5 minutes (using cached data)

---

## 🚀 Quick Start

### One-Line Setup

```bash
# Clone repository
git clone https://github.com/qingsongcui/housing-narratives.git
cd housing-narratives

# Install dependencies
pip install -r requirements.txt

# Run full analysis
python code/run_all.py
```

### Manual Execution

If you prefer to run scripts individually:

```bash
# Step 1: Data pipeline (skip if using cached data)
python code/03_real_data_pipeline.py

# Step 2: Main analysis
python code/04_analysis_real.py

# Step 3: Mechanism tests
python code/05_interaction_models.py
python code/07_mechanism_saiz.py

# Step 4: Generate tables
python code/12_final_tables.py
```

---

## 📁 Repository Structure

```
housing-narratives/
├── README.md                   # This file
├── REPLICATION.md              # Detailed replication guide
├── CITATION.cff                # Citation metadata
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── code/                       # Analysis scripts
│   ├── constants.py            # Sample statistics (single source of truth)
│   ├── 03_real_data_pipeline.py    # Data construction
│   ├── 04_analysis_real.py         # Main regression analysis
│   ├── 05_interaction_models.py    # Mechanism: narrative × friction
│   ├── 06_discriminant_validity.py # Robustness checks
│   ├── 07_mechanism_saiz.py        # Mechanism: Saiz supply elasticity
│   ├── 11_wild_bootstrap.py        # Statistical inference
│   ├── 12_final_tables.py          # Generate LaTeX tables
│   └── run_all.py                  # Master script (run all analyses)
│
├── data/                       # Data directory
│   ├── raw/                    # Original, immutable data
│   │   ├── redfin_metro.tsv.gz
│   │   ├── fred_cache/
│   │   │   ├── MORTGAGE30US.csv
│   │   │   └── UNRATE.csv
│   │   └── trends_cache/
│   │       └── dma_trends_quarterly.csv
│   ├── processed/              # Cleaned/processed data
│   │   └── panel_data_real.csv
│   └── mappings/               # Crosswalk files
│       └── metro_dma_crosswalk_deterministic.csv
│
├── output/                     # Generated outputs
│   ├── figures/                # Main paper figures
│   └── tables/                 # Regression tables (LaTeX)
│
├── paper/                      # LaTeX source files
│   ├── main.tex                # Main manuscript
│   ├── references.bib          # Bibliography
│   └── sections/               # Section files
│
└── docs/                       # Documentation
    └── DATA_PIPELINE.md        # Detailed data guide
```

---

## 📚 Documentation

- **[REPLICATION.md](REPLICATION.md)** - Step-by-step replication guide with code-to-paper mapping
- **[DATA_PIPELINE.md](docs/DATA_PIPELINE.md)** - Detailed data construction documentation
- **[COVER_LETTER.md](COVER_LETTER.md)** - Journal submission cover letter

---

## 🔬 Citation

If you use this code or data, please cite:

```bibtex
@article{cui2025friction,
  title={The Friction Gate: Narrative Transmission in Housing Markets},
  author={Cui, Qingsong},
  year={2025},
  journal={Journal of Economic Behavior \& Organization},
  note={Under Review}
}
```

---

## 📧 Contact

**Qingsong Cui**
- Email: qingsongcui9857@gmail.com
- GitHub: [@qingsongcui](https://github.com/qingsongcui)

For questions about the replication, please open an issue on GitHub.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Data provided by Redfin and the Federal Reserve Economic Data (FRED)
- Google Trends data accessed via pytrends
- Research conducted at [Your Institution]

---

*Last updated: February 2025*
