# 数据重跑与复现指南

## 1. 目标
本文档指导你如何完整重跑 `code/03_real_data_pipeline.py`，确保论文中的面板数据可复现、可提交 “稳 Q1 期刊” 所需的高质量版本。主要工作流如下：

1. **准备原始数据**（Redfin、FRED、Google Trends、跨区映射）。
2. **运行 `03_real_data_pipeline.py`** 生成 `data/processed/panel_data_real.csv`。
3. **使用 crosswalk 脚本**（如 `code/finalize_crosswalk.py`）确认 143 个 Metro → DMA 映射。

## 2. 原始数据需求

### 2.1 Redfin Metro 数据（已包含）
- 当前仓库带有 `data/raw/redfin_metro.tsv.gz`，它即 Redfin Metro Market Tracker 的压缩版。若想更新，请从：
  `https://www.redfin.com/news/data-center/` 下载最新的 `redfin_metro_market_tracker.tsv000.gz`，并压缩替换同名文件。

### 2.2 FRED 宏观变量
脚本需要下列季度数据：
| 系列 | 说明 | 推荐保存路径 |
| --- | --- | --- |
| `MORTGAGE30US` | 30 年固定利率抵押贷款 | `data/raw/fred_cache/MORTGAGE30US.csv` |
| `UNRATE` | 失业率 | `data/raw/fred_cache/UNRATE.csv` |

每个 CSV 文件必须包含 `DATE` 列（格式 `YYYY-MM-DD`）和对应指标列。下载方式：

```bash
curl -o data/raw/fred_cache/MORTGAGE30US.csv \
  https://fred.stlouisfed.org/series/MORTGAGE30US/downloaddata/MORTGAGE30US.csv
curl -o data/raw/fred_cache/UNRATE.csv \
  https://fred.stlouisfed.org/series/UNRATE/downloaddata/UNRATE.csv
```

确保目录存在：`mkdir -p data/raw/fred_cache`。

### 2.3 Google Trends Narrative（DMA 级别）
Narrative 指数需聚合下列关键词：

| Buy Narrative | Risk Narrative |
|---------------|----------------|
| buy a house | housing crash |
| homes for sale | foreclosure |
| mortgage preapproval | mortgage rate |
| first time home buyer | house price bubble |
| down payment | recession |

请使用 `pytrends` 分别对每个匹配的 DMA（`code/03_real_data_pipeline.py` 中的 `METRO_DMA_CROSSWALK`）抓取 2012Q1–2024Q4 的月度数据；脚本会在 `data/raw/trends_cache/` 下缓存经过季度化处理的 CSV。也可手动准备格式如下（最少包含 `dma_code`、`quarter`、`buy_*`、`risk_*` 列）并保存为：
```
data/raw/trends_cache/dma_trends_quarterly.csv
```

确保该 CSV 采样频率为季度，并且 `quarter` 列可被 `pd.to_datetime` 识别（例如 `2012-03-31`）。

### 2.4 Crosswalk 映射
脚本默认使用 `code/finalize_crosswalk.py` 生成的可审计表 `data/mappings/metro_dma_crosswalk_deterministic.csv`。若需验证或更新映射，请按顺序运行：
1. `python code/rebuild_crosswalk.py`
2. `python code/finalize_crosswalk.py`

## 3. 运行顺序

```bash
python -m pip install -r requirements.txt  # 确保 requests、pytrends、pandas、numpy 已安装
python code/rebuild_crosswalk.py           # 可选：重建 deterministic crosswalk
python code/03_real_data_pipeline.py
```

`03_real_data_pipeline.py` 会：
1. 分析 Redfin、DMA、季度汇总；
2. 采集/缓存 FRED 数据（如果已有缓存则跳过下载）；
3. 采集/缓存 Google Trends；
4. 生成标准化 narrative、滞后变量和其他控制变量，输出 `data/processed/panel_data_real.csv`。

若脚本报告缺少缓存（例如 FRED 或 Trends 访问失败），请按第 2 节提前准备好 CSV，再重试。

## 4. 缓存与调试要点

- FRED 缓存目录：`data/raw/fred_cache/*.csv`，脚本会自动写入下载过的内容。
- Trends 缓存文件：`data/raw/trends_cache/dma_trends_quarterly.csv`，每次成功抓取后会覆盖更新。
- 关键日志：若看到 “Critical Error” 信息，说明某个数据源既不能在线拉取，也没有对应缓存。
- 若需手动拼接数据，可使用 `pandas` 从各 DMA 的 monthly series 汇总到 quarterly，然后按章节描述的标准化方式构造 `n_buy` / `n_risk`。

## 5. 复现验证

1. 确保 `python code/04_analysis_real.py` 能加载新生成的 `panel_data_real.csv` 且没有 `KeyError`。
2. 运行 `python code/12_final_tables.py` 检查基础回归表是否能生成。
3. 对照 `paper/sections/data.tex`、`paper/appendix/data_dictionary.tex` 中的样本统计，确认 `constants.py` 中的数字仍然准确。
