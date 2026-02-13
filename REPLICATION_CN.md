# 详细复现指南（中文版）

本文件提供论文《摩擦之门：住房市场叙事传播》结果的全面复现指南。

---

## 目录

1. [计算环境](#计算环境)
2. [数据管道](#数据管道)
3. [分析工作流](#分析工作流)
4. [代码到论文映射](#代码到论文映射)
5. [预期输出](#预期输出)
6. [故障排除](#故障排除)

---

## 计算环境

### 硬件要求

- **CPU**：现代多核处理器（Intel i5/AMD Ryzen 5或更好）
- **内存**：最低8GB，建议16GB
- **存储**：约500MB可用空间
- **网络**：仅在刷新Google Trends数据时需要

### 软件要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.9+ | 已在3.9、3.10、3.11上测试 |
| Pandas | 1.5+ | 数据处理 |
| NumPy | 1.21+ | 数值计算 |
| Matplotlib | 3.5+ | 可视化 |
| LinearModels | 4.27+ | 面板回归 |

### 环境设置

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import pandas, numpy, matplotlib, linearmodels; print('所有包已安装')"
```

---

## 数据管道

### 概述

数据管道从三个主要来源构建美国住房市场的季度面板：

1. **Redfin Metro Market Tracker** - 住房交易量、价格、库存
2. **FRED** - 宏观经济控制变量（抵押贷款利率、失业率）
3. **Google Trends** - DMA级别的叙事注意力指数

### 管道步骤

#### 步骤1：交叉映射构建（可选）

**脚本**：`code/finalize_crosswalk.py`

这将创建201个Redfin大都市与127个Nielsen DMA之间的确定性映射。

```bash
python code/finalize_crosswalk.py
```

**输出**：`data/mappings/metro_dma_crosswalk_deterministic.csv`

> **注意**：此步骤是可选的。交叉映射已包含在仓库中。

#### 步骤2：主数据管道

**脚本**：`code/03_real_data_pipeline.py`

这是主要的数据构建脚本。它：

1. 加载Redfin住房数据（约108MB压缩）
2. 获取/缓存FRED宏观变量
3. 获取/缓存Google Trends叙事指数
4. 构建季度面板
5. 计算标准化的叙事指数
6. 生成滞后变量和控制变量

```bash
# 使用缓存数据（复现推荐）
python code/03_real_data_pipeline.py

# 强制刷新Trends数据（需要网络，约30分钟）
# 编辑脚本设置 USE_CACHED_TRENDS = False
```

**输入文件**：
- `data/raw/redfin_metro.tsv.gz`（已包含）
- `data/raw/fred_cache/*.csv`（已包含）
- `data/raw/trends_cache/dma_trends_quarterly.csv`（已包含）

**输出**：`data/processed/panel_data_real.csv`

**预期控制台输出**：
```
Loading Redfin data...
Processing quarterly panel...
Fetching FRED data (cached)...
Loading Google Trends (cached)...
Final sample: 6984 observations, 201 metros, 127 DMAs
Panel saved to: data/processed/panel_data_real.csv
```

### 数据字典

| 变量 | 说明 | 来源 |
|------|------|------|
| `region` | 大都市区名称 | Redfin |
| `dma_code` | Nielsen DMA标识符 | 交叉映射 |
| `quarter` | 季度（YYYY-MM-DD） | 构建 |
| `volume_growth` | Δln(交易量) | Redfin |
| `n_buy` | 买入叙事指数（z标准化） | Google Trends |
| `n_risk` | 风险叙事指数（z标准化） | Google Trends |
| `mortgage_rate` | 30年固定抵押贷款利率 | FRED |
| `unemployment` | 失业率 | FRED |
| `jumbo_exposure` | 高价值交易份额 | Redfin |

---

## 分析工作流

### 快速运行（推荐）

```bash
python code/run_all.py
```

此主脚本按正确顺序执行所有分析并生成所有输出。

### 手动执行顺序

如果单独运行脚本，请按此顺序执行：

#### 1. 基线分析
**脚本**：`code/04_analysis_real.py`

运行表2（基线结果）的主OLS和面板回归。

```bash
python code/04_analysis_real.py
```

**关键输出**：
- 控制台：系数估计、R²、标准误
- 文件：`output/regression_results_real.txt`

**预期结果**（Phase 1.8B确定性交叉映射）：
```
BASELINE RESULTS
================
n_buy coefficient: -0.0326 (SE: 0.017, p=0.055)
n_risk coefficient: -0.0061
Valuation-Volume Ratio: 10.28
```

#### 2. 机制：叙事 × 摩擦交互
**脚本**：`code/05_interaction_models.py`

检验"摩擦之门"假设：基于库存约束的条件叙事效应。

```bash
python code/05_interaction_models.py
```

**关键输出**：
- 交互系数（n_buy × low_inventory）
- 按库存三分位的子样本分析

#### 3. 机制：供应弹性（Saiz）
**脚本**：`code/07_mechanism_saiz.py`

按住房供应弹性检验叙事效应（Saiz, 2010）。

```bash
python code/07_mechanism_saiz.py
```

**关键输出**：
- 按弹性四分位的分析
- 高摩擦vs低摩擦子样本

#### 4. 稳健性检验
**脚本**：
- `code/06_discriminant_validity.py` - 安慰剂检验
- `code/07_robustness.py` - 替代规范
- `code/11_wild_bootstrap.py` - Wild bootstrap推断

```bash
python code/06_discriminant_validity.py
python code/07_robustness.py
python code/11_wild_bootstrap.py
```

#### 5. 生成表格
**脚本**：`code/12_final_tables.py`

为论文创建LaTeX格式的回归表格。

```bash
python code/12_final_tables.py
```

**输出**：`output/tables/`目录中的`.tex`文件

---

## 代码到论文映射

### 主要结果

| 论文元素 | 脚本 | 输出文件 | 行/页 |
|---------|------|---------|-------|
| **表2**：基线面板回归 | `04_analysis_real.py` | `output/regression_results_real.txt` | 表2 |
| **表3**：机制 - 库存 | `05_interaction_models.py` | 控制台输出 | 表3 |
| **表4**：机制 - 供应弹性 | `07_mechanism_saiz.py` | 控制台输出 | 表4 |
| **图2**：叙事时间序列 | `04_analysis_real.py` | `output/figures/narrative_series.pdf` | 图2 |
| **图3**：交互效应 | `05_interaction_models.py` | `output/figures/interaction_plot.pdf` | 图3 |

### 稳健性和附录

| 论文元素 | 脚本 | 说明 |
|---------|------|------|
| 附录A：数据构建 | `03_real_data_pipeline.py` | 完整管道文档 |
| 附录B：判别效度 | `06_discriminant_validity.py` | 安慰剂检验 |
| 附录C：替代规范 | `07_robustness.py` | 稳健性检验 |
| 附录D：Wild Bootstrap | `11_wild_bootstrap.py` | 聚类误差推断 |

### 样本统计

所有样本统计都定义在`code/constants.py`中，应匹配：

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

## 预期输出

### 复现后的目录结构

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

### 验证清单

运行复现后，验证：

- [ ] `panel_data_real.csv`存在于`data/processed/`
- [ ] 控制台显示："Final sample: 6984 observations"
- [ ] 表2基线系数：n_buy ≈ -0.033（SE ≈ 0.017）
- [ ] VP比率报告为~10.3
- [ ] 所有图表生成于`output/figures/`

---

## 故障排除

### 问题：缺失数据文件

**症状**：`FileNotFoundError: panel_data_real.csv not found`

**解决方案**：
```bash
# 首先运行数据管道
python code/03_real_data_pipeline.py
```

### 问题：Google Trends速率限制

**症状**：`TooManyRequestsError`来自pytrends

**解决方案**：仓库包含缓存的Trends数据。如果刷新：
```python
# 在03_real_data_pipeline.py中，请求之间添加延迟
import time
time.sleep(5)  # 在API调用之间添加
```

### 问题：加载Redfin数据时内存错误

**症状**：加载Redfin数据时`MemoryError`

**解决方案**：
```python
# 在03_real_data_pipeline.py中使用分块
df = pd.read_csv('redfin_metro.tsv.gz', compression='gzip', chunksize=100000)
```

### 问题：LaTeX编译错误

**症状**：表格中出现`! Undefined control sequence`

**解决方案**：确保安装所需的LaTeX包：
```bash
# 在LaTeX前言中
\usepackage{booktabs}
\usepackage{threeparttable}
\usepackage{siunitx}
```

### 问题：与论文结果不同

**检查清单**：
1. [ ] 使用确定性交叉映射？（`metro_dma_crosswalk_deterministic.csv`）
2. [ ] 完整案例分析（无插补）？
3. [ ] DMA级别聚类？（不是大都市级别）
4. [ ] 正确的时间段（2012Q1-2024Q4）？
5. [ ] 变量在关键词内标准化？

如果问题仍然存在，请检查`code/constants.py`中的验证统计。

---

## 联系

如有本指南未解决的复现问题，请：

1. 检查现有的[GitHub Issues](https://github.com/qingsongcui/housing-narratives/issues)
2. 提交新问题并提供：
   - Python版本（`python --version`）
   - 错误信息（完整堆栈跟踪）
   - 操作系统
   - 重现步骤

---

*最后更新：2025年2月*
