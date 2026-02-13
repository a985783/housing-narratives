"""
ECR-CAS Housing Cycles - REAL Data Pipeline (DMA-Level)
========================================================
This script implements a robust pipeline to acquire REAL data:
1. Redfin Metro Data (Full dataset, filtered for Top 300)
2. Google Trends (DMA-level granularity for proper metro identification)
3. FRED Data (Mortgage Rates, Unemployment)
4. Merge and Process into Panel Data

IMPORTANT: This version fetches DMA-level Trends to avoid the issue
of all metros in the same state sharing the same narrative index.
"""

import pandas as pd
import numpy as np
import requests
import io
import time
import os
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

FRED_CACHE_DIR = os.path.join(DATA_DIR, "fred_cache")
TRENDS_CACHE_DIR = os.path.join(DATA_DIR, "trends_cache")
TRENDS_CACHE_FILE = os.path.join(TRENDS_CACHE_DIR, "dma_trends_quarterly.csv")

KEYWORDS_BUY = [
    "buy a house",
    "homes for sale",
    "mortgage preapproval",
    "first time home buyer",
    "down payment",
]

KEYWORDS_RISK = [
    "housing crash",
    "foreclosure",
    "mortgage rate",
    "house price bubble",
    "recession",
]

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# =============================================================================
# METRO TO DMA CROSSWALK
# =============================================================================
# Top 50 metros with their corresponding Google Trends DMA codes
# DMA codes are from Nielsen's Designated Market Areas
# Format: 'Metro Name Pattern': 'US-ST-###' (state-dma_code)
# =============================================================================
# METRO TO DMA CROSSWALK (~150 Metros / Top 90% Volume)
# =============================================================================
# Format: 'Metro Name Pattern': 'US-ST-###' (state-dma_code)
# Mappings based on Nielsen DMA boundaries. Satellites mapped to primary DMA.
METRO_DMA_CROSSWALK = {
    # 1. Northeast
    'New York': 'US-NY-501',
    'Newark': 'US-NY-501',
    'Jersey City': 'US-NY-501',
    'Nassau': 'US-NY-501',
    'Bridgeport': 'US-NY-501',
    'Stamford': 'US-NY-501',
    'Philadelphia': 'US-PA-504',
    'Camden': 'US-PA-504',
    'Wilmington': 'US-PA-504',
    'Allentown': 'US-PA-504',
    'Boston': 'US-MA-506',
    'Cambridge': 'US-MA-506',
    'Newton': 'US-MA-506',
    'Worcester': 'US-MA-506',
    'Providence': 'US-RI-521',
    'Hartford': 'US-CT-533',
    'New Haven': 'US-CT-533',
    'Pittsburgh': 'US-PA-508',
    'Buffalo': 'US-NY-514',
    'Rochester': 'US-NY-538',
    'Albany': 'US-NY-532',
    
    # 2. Midwest
    'Chicago': 'US-IL-602',
    'Naperville': 'US-IL-602',
    'Elgin': 'US-IL-602',
    'Detroit': 'US-MI-505',
    'Warren': 'US-MI-505',
    'Ann Arbor': 'US-MI-505',
    'Minneapolis': 'US-MN-613',
    'St. Paul': 'US-MN-613',
    'Cleveland': 'US-OH-510',
    'Columbus': 'US-OH-535',
    'Cincinnati': 'US-OH-515',
    'Indianapolis': 'US-IN-527',
    'St. Louis': 'US-MO-609',
    'Kansas City': 'US-MO-616',
    'Milwaukee': 'US-WI-617',
    'Grand Rapids': 'US-MI-563',
    'Oklahoma City': 'US-OK-650',
    'Tulsa': 'US-OK-671',
    'Omaha': 'US-NE-740',
    'Des Moines': 'US-IA-679',
    
    # 3. South
    'Washington': 'US-DC-511',
    'Arlington': 'US-DC-511',
    'Alexandria': 'US-DC-511',
    'Baltimore': 'US-MD-512',
    'Atlanta': 'US-GA-524',
    'Sandy Springs': 'US-GA-524',
    'Miami': 'US-FL-528',
    'Fort Lauderdale': 'US-FL-528',
    'West Palm Beach': 'US-FL-548',
    'Tampa': 'US-FL-539',
    'St. Petersburg': 'US-FL-539',
    'Orlando': 'US-FL-534',
    'Jacksonville': 'US-FL-561',
    'Charlotte': 'US-NC-517',
    'Raleigh': 'US-NC-560',
    'Durham': 'US-NC-560',
    'Nashville': 'US-TN-659',
    'Memphis': 'US-TN-640',
    'New Orleans': 'US-LA-622',
    'Louisville': 'US-KY-529',
    'Birmingham': 'US-AL-630',
    'Richmond': 'US-VA-556',
    'Virginia Beach': 'US-VA-544',
    'Norfolk': 'US-VA-544',
    'Greensboro': 'US-NC-518',
    'Knoxville': 'US-TN-557',
    'Greenville': 'US-SC-567',
    'Columbia': 'US-SC-546',
    'Charleston': 'US-SC-519',
    
    # 4. Texas / Southwest
    'Dallas': 'US-TX-623',
    'Fort Worth': 'US-TX-623',
    'Arlington': 'US-TX-623',
    'Plano': 'US-TX-623',
    'Houston': 'US-TX-618',
    'The Woodlands': 'US-TX-618',
    'San Antonio': 'US-TX-641',
    'Austin': 'US-TX-635',
    'El Paso': 'US-TX-765',
    'McAllen': 'US-TX-636',
    'Phoenix': 'US-AZ-753',
    'Mesa': 'US-AZ-753',
    'Scottsdale': 'US-AZ-753',
    'Tucson': 'US-AZ-789',
    'Albuquerque': 'US-NM-790',
    'Las Vegas': 'US-NV-839',
    
    # 5. West
    'Los Angeles': 'US-CA-803',
    'Long Beach': 'US-CA-803',
    'Anaheim': 'US-CA-803',
    'Riverside': 'US-CA-803',
    'San Bernardino': 'US-CA-803',
    'San Francisco': 'US-CA-807',
    'Oakland': 'US-CA-807',
    'Berkeley': 'US-CA-807',
    'San Jose': 'US-CA-807',
    'Sunnyvale': 'US-CA-807',
    'Santa Clara': 'US-CA-807',
    'San Diego': 'US-CA-825',
    'Sacramento': 'US-CA-862',
    'Fresno': 'US-CA-866',
    'Bakersfield': 'US-CA-800',
    'Seattle': 'US-WA-819',
    'Tacoma': 'US-WA-819',
    'Bellevue': 'US-WA-819',
    'Portland': 'US-OR-820',
    'Vancouver': 'US-OR-820',
    'Denver': 'US-CO-751',
    'Aurora': 'US-CO-751',
    'Salt Lake City': 'US-UT-770',
    'Honolulu': 'US-HI-744',
    'Boise': 'US-ID-757',
    'Spokane': 'US-WA-881',
}

def get_dma_for_metro(metro_name):
    """Match a metro name to its DMA code using fuzzy matching."""
    for pattern, dma_code in METRO_DMA_CROSSWALK.items():
        if pattern.lower() in metro_name.lower():
            return dma_code
    return None


def load_cached_fred(series_id):
    cache_path = os.path.join(FRED_CACHE_DIR, f"{series_id}.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        # Handle both DATE and observation_date column names
        date_col = 'DATE' if 'DATE' in df.columns else 'observation_date'
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.rename(columns={date_col: 'DATE'})
        df = df.set_index("DATE")
        return df
    return None


def load_cached_trends():
    if os.path.exists(TRENDS_CACHE_FILE):
        df = pd.read_csv(TRENDS_CACHE_FILE, parse_dates=["quarter"])
        return df
    return None
# =============================================================================
# 1. REDFIN METRO DATA
# =============================================================================
def fetch_redfin_metro():
    print("\n[1/4] Fetching Redfin METRO Data...")
    url = "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/redfin_metro_market_tracker.tsv000.gz"
    local_path = os.path.join(DATA_DIR, "redfin_metro.tsv.gz")
    
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 1000000:
        print(f"  Downloading from {url}...")
        try:
            r = requests.get(url, stream=True)
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("  Download complete.")
        except Exception as e:
            print(f"  Error downloading: {e}")
            return None, None
    else:
        print(f"  Using existing local file at {local_path}")
        
    print("  Loading and filtering...")
    try:
        cols = ['REGION', 'PERIOD_BEGIN', 'HOMES_SOLD', 'MEDIAN_SALE_PRICE', 
                'INVENTORY', 'STATE_CODE', 'PROPERTY_TYPE']
        
        chunks = []
        for chunk in pd.read_csv(local_path, sep='\t', compression='gzip', usecols=cols, chunksize=50000):
            chunk['PERIOD_BEGIN'] = pd.to_datetime(chunk['PERIOD_BEGIN'])
            # Filter: All Residential, 2012Q1 - 2024Q4
            mask = ((chunk['PROPERTY_TYPE'] == 'All Residential') & 
                    (chunk['PERIOD_BEGIN'] >= '2012-01-01') &
                    (chunk['PERIOD_BEGIN'] < '2025-01-01'))
            chunks.append(chunk[mask])
            
        df = pd.concat(chunks, ignore_index=True)
        print(f"  Loaded {len(df)} rows of residential data.")
        
        # Identify Top 300 by volume
        volume_rank = df.groupby('REGION')['HOMES_SOLD'].sum().sort_values(ascending=False)
        top_300_metros = volume_rank.head(300).index.tolist()
        
        df_filtered = df[df['REGION'].isin(top_300_metros)].copy()
        
        # Rename
        df_filtered = df_filtered.rename(columns={
            'REGION': 'region', 
            'HOMES_SOLD': 'homes_sold',
            'MEDIAN_SALE_PRICE': 'median_sale_price',
            'INVENTORY': 'inventory',
            'STATE_CODE': 'state',
            'PERIOD_BEGIN': 'period_begin'
        })
        
        # Add DMA mapping
        df_filtered['dma_code'] = df_filtered['region'].apply(get_dma_for_metro)
        # Fill NaN dma_code with 'UNKNOWN' to preserve them during groupby
        df_filtered['dma_code'] = df_filtered['dma_code'].fillna('UNKNOWN')
        
        dma_coverage = (df_filtered['dma_code'] != 'UNKNOWN').mean()
        print(f"  DMA coverage: {dma_coverage:.1%} of rows mapped to DMAs")
        
        # Aggregate to quarterly
        df_filtered['quarter'] = df_filtered['period_begin'].dt.to_period('Q').dt.to_timestamp()
        
        df_q = df_filtered.groupby(['region', 'state', 'dma_code', 'quarter']).agg({
             'homes_sold': 'sum',
             'median_sale_price': 'mean',
             'inventory': 'mean'
        }).reset_index()
        
        # Extract unique DMAs for Trends fetching
        unique_dmas = [d for d in df_q['dma_code'].unique().tolist() if d != 'UNKNOWN']
        
        print(f"  Quarterly obs: {len(df_q)}, Metros: {df_filtered['region'].nunique()}, DMAs: {len(unique_dmas)}")
        return df_q, unique_dmas
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback; traceback.print_exc()
        return None, None

# =============================================================================
# 2. FRED DATA
# =============================================================================
def fetch_fred_series(series_id, name):
    print(f"  Fetching {name} ({series_id})...")
    url = f"https://fred.stlouisfed.org/series/{series_id}/downloaddata/{series_id}.csv"
    cache_path = os.path.join(FRED_CACHE_DIR, f"{series_id}.csv")
    cached = load_cached_fred(series_id)
    if cached is not None:
        print(f"  Loaded cached {name} from {cache_path}")
        df_q = cached.resample('QS').mean()
        return df_q

    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")

        df = pd.read_csv(io.StringIO(r.text))
        if 'DATE' not in df.columns:
            raise RuntimeError("FRED response missing DATE column")

        df['DATE'] = pd.to_datetime(df['DATE'])
        val_col = [c for c in df.columns if c != 'DATE'][0]
        df = df.rename(columns={val_col: series_id})
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')

        ensure_dir(FRED_CACHE_DIR)
        df.to_csv(cache_path, index=False)
        print(f"  Cached {name} to {cache_path}")

        df = df.set_index('DATE')
        df_q = df.resample('QS').mean()
        return df_q

    except Exception as e:
        raise RuntimeError(
            f"{name} ({series_id}) 无法下载。请检查网络或将 CSV 放在 {cache_path}，"
            f"下载地址 https://fred.stlouisfed.org/series/{series_id}/downloaddata/{series_id}.csv。原始错误: {e}"
        ) from e

def fetch_macro_data():
    print("\n[2/4] Fetching FRED Macro Data...")
    mtg = fetch_fred_series('MORTGAGE30US', 'Mortgage Rate')
    unrate = fetch_fred_series('UNRATE', 'Unemployment Rate')

    dates = pd.date_range(start='2012-01-01', end='2024-10-01', freq='QS')
    df_macro = pd.DataFrame(index=dates)
    df_macro = df_macro.join(mtg, how='left')
    df_macro = df_macro.join(unrate, how='left')

    df_macro['mortgage_rate'] = df_macro['MORTGAGE30US']
    df_macro['unemployment'] = df_macro['UNRATE']
    df_macro = df_macro[['mortgage_rate', 'unemployment']]

    df_macro = df_macro.interpolate().ffill().bfill()
    print(f"  Macro data: {len(df_macro)} quarters")
    return df_macro

# =============================================================================
# 3. GOOGLE TRENDS (DMA LEVEL)
# =============================================================================
def fetch_dma_weights(keywords):
    """
    Fetch cross-sectional weights for DMAs using interest_by_region.
    Returns a dict mapping dma_code (str) -> weight (float).
    """
    print(f"  Fetching cross-sectional weights for {keywords}...")
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
        
        # We need a long timeframe to get a stable "average interest"
        pytrends.build_payload(keywords, timeframe='2012-01-01 2024-12-31', geo='US')
        
        # Fetch DMA resolution
        # inc_geo_code=True returns numeric codes like 501
        df_weights = pytrends.interest_by_region(resolution='DMA', inc_geo_code=True, inc_low_vol=True)
        
        if df_weights.empty:
            print("  Warning: Empty weights returned.")
            return None

        # Calculate average weight across keywords (if multiple)
        # The result columns are: geoCode, keyword1, keyword2...
        kw_cols = [c for c in df_weights.columns if c != 'geoCode']
        df_weights['mean_weight'] = df_weights[kw_cols].mean(axis=1)
        
        # Create map: "501" -> weight
        weight_map = df_weights.set_index('geoCode')['mean_weight'].to_dict()
        
        print(f"  Retrieved weights for {len(weight_map)} DMAs.")
        return weight_map
        
    except Exception as e:
        print(f"  Error fetching weights: {e}")
        return None

def fetch_dma_trends(dma_codes, keywords_buy, keywords_risk):
    print(f"\n[3/4] Fetching Google Trends (DMA Level) for {len(dma_codes)} DMAs...")
    
    # 1. Fetch Weights (Cross-Sectional Scale)
    print("  Step 3a: Fetching Reference Weights (interest_by_region)...")
    buy_weights = fetch_dma_weights(keywords_buy)
    risk_weights = fetch_dma_weights(keywords_risk)
    
    cached = load_cached_trends()
    if cached is not None:
        print(f"  Loaded cached DMA trends from {TRENDS_CACHE_FILE}")
        return cached

    try:
        from pytrends.request import TrendReq
    except ImportError as e:
        raise RuntimeError(
            "pytrends 未安装。请运行 `pip install pytrends` 或准备好缓存文件后重试。"
        ) from e

    try:
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
    except Exception as e:
        raise RuntimeError(
            f"pytrends 初始化失败：{e}。请检查网络或代理设置。"
        ) from e

    all_trends = []
    success_count = 0
    
    for dma_code in dma_codes:
        # Extract numeric code from 'US-NY-501' -> '501'
        try:
            numeric_code = dma_code.split('-')[-1]
        except:
            numeric_code = '000'
            
        print(f"  Fetching {dma_code}...", end='', flush=True)
        
        try:
            # Fetch Buy keywords
            pytrends.build_payload(keywords_buy, timeframe='2012-01-01 2024-12-31', geo=dma_code)
            df_buy = pytrends.interest_over_time()
            
            if df_buy.empty:
                print(" ✗ (Empty)")
                continue
                
            df_buy = df_buy.drop(columns=['isPartial'], errors='ignore')
            
            # SCALING FIX: Apply cross-sectional weight
            if buy_weights and str(numeric_code) in buy_weights:
                target_mean = buy_weights[str(numeric_code)]
                current_mean = df_buy.values.mean()
                if current_mean > 0:
                    scale_factor = target_mean / current_mean
                    df_buy = df_buy * scale_factor
            
            df_buy.columns = [f'buy_{c.replace(" ","_")}' for c in df_buy.columns]
            
            time.sleep(1)  # Rate limit protection
            
            # Fetch Risk keywords
            pytrends.build_payload(keywords_risk, timeframe='2012-01-01 2024-12-31', geo=dma_code)
            df_risk = pytrends.interest_over_time()
            
            if not df_risk.empty:
                df_risk = df_risk.drop(columns=['isPartial'], errors='ignore')
                
                # SCALING FIX: Risk
                if risk_weights and str(numeric_code) in risk_weights:
                    target_mean = risk_weights[str(numeric_code)]
                    current_mean = df_risk.values.mean()
                    if current_mean > 0:
                        scale_factor = target_mean / current_mean
                        df_risk = df_risk * scale_factor
                
                df_risk.columns = [f'risk_{c.replace(" ","_")}' for c in df_risk.columns]
                
            # Merge
            df_dma = pd.concat([df_buy, df_risk], axis=1)
            df_dma['dma_code'] = dma_code
            df_dma['date'] = df_dma.index
            all_trends.append(df_dma)
            success_count += 1
            print(" ✓")
            
            time.sleep(1)
            
        except Exception as e:
            print(f" ✗ Error: {str(e)[:50]}")
            time.sleep(5)  # Backoff
            
    print(f"  Successfully fetched {success_count}/{len(dma_codes)} DMAs")

    if not all_trends:
        raise RuntimeError(
            "未能从 Google Trends 获取任何 DMA 数据。请检查网络/pytrends，"
            f"或预先准备 {TRENDS_CACHE_FILE}。"
        )

    full_trends = pd.concat(all_trends)
    full_trends['quarter'] = pd.to_datetime(full_trends['date']).dt.to_period('Q').dt.to_timestamp()

    quarterly_trends = full_trends.groupby(['dma_code', 'quarter']).mean(numeric_only=True).reset_index()

    ensure_dir(TRENDS_CACHE_DIR)
    quarterly_trends.to_csv(TRENDS_CACHE_FILE, index=False)
    print(f"  Cached DMA trends to {TRENDS_CACHE_FILE}")

    return quarterly_trends

# =============================================================================
# 4. MASTER MERGE
# =============================================================================
def main():
    print("=" * 60)
    print("REAL DATA PIPELINE (DMA-Level Narrative)")
    print("=" * 60)
    ensure_dir(FRED_CACHE_DIR)
    ensure_dir(TRENDS_CACHE_DIR)
    
    # 1. Redfin
    df_redfin, unique_dmas = fetch_redfin_metro()
    if df_redfin is None:
        print("Critical Error: Redfin download failed.")
        return
        
    # 2. FRED
    try:
        df_macro = fetch_macro_data()
    except RuntimeError as e:
        print(f"Critical Error: {e}")
        return
        
    # 3. Trends (DMA Level)
    try:
        df_trends = fetch_dma_trends(unique_dmas, KEYWORDS_BUY, KEYWORDS_RISK)
    except RuntimeError as e:
        print(f"Critical Error: {e}")
        return
    
    # 4. Merge
    print("\n[4/4] Merging Datasets...")
    
    # Merge Redfin + Trends (on DMA + Quarter)
    if df_trends is not None:
        # Ensure dma_code is same type in both dataframes
        df_redfin['dma_code'] = df_redfin['dma_code'].astype(str)
        df_trends['dma_code'] = df_trends['dma_code'].astype(str)
        df_merged = pd.merge(df_redfin, df_trends, on=['dma_code', 'quarter'], how='left')
        missing_pct = df_merged[[c for c in df_merged.columns if c in ['buy_a_house', 'homes_for_sale']]].isna().mean().mean()
        print(f"  Narrative missing rate after DMA merge: {missing_pct:.1%}")
    else:
        df_merged = df_redfin
        print("  Warning: Trends missing, using fallback.")
        
    # Merge + Macro (on Quarter)
    df_macro['quarter'] = df_macro.index
    df_final = pd.merge(df_merged, df_macro, on='quarter', how='left')
    
    # Calculate Growth Rates
    df_final['ln_volume'] = np.log(df_final['homes_sold'].clip(lower=1))
    df_final = df_final.sort_values(['region', 'quarter'])
    df_final['volume_growth'] = df_final.groupby('region')['ln_volume'].diff()
    
    df_final['ln_price'] = np.log(df_final['median_sale_price'].clip(lower=1))
    df_final['price_growth'] = df_final.groupby('region')['ln_price'].diff()
    
    # Create Standardized Narrative Indices
    if df_trends is not None:
        # Column names from cache: buy_a_house, homes_for_sale, etc.
        # Adjusted to match prefix added in fetch_dma_trends
        buy_keywords = ['buy_buy_a_house', 'buy_homes_for_sale', 'buy_mortgage_preapproval', 
                        'buy_first_time_home_buyer', 'buy_down_payment']
        risk_keywords = ['risk_housing_crash', 'risk_foreclosure', 'risk_mortgage_rate', 
                         'risk_house_price_bubble', 'risk_recession']
        
        buy_cols = [c for c in buy_keywords if c in df_final.columns]
        risk_cols = [c for c in risk_keywords if c in df_final.columns]
        
        print(f"  Found buy columns: {buy_cols}")
        print(f"  Found risk columns: {risk_cols}")
        
        if buy_cols:
            # Standardize within-sample
            for col in buy_cols + risk_cols:
                if col in df_final.columns:
                    mean_val = df_final[col].mean()
                    std_val = df_final[col].std()
                    if std_val > 0:
                        df_final[col] = (df_final[col] - mean_val) / std_val
            
            df_final['n_buy'] = df_final[buy_cols].mean(axis=1)
            df_final['n_risk'] = df_final[risk_cols].mean(axis=1) if risk_cols else 0
            for col in ['n_buy', 'n_risk']:
                if col in df_final.columns:
                    mean_val = df_final[col].mean()
                    std_val = df_final[col].std()
                    if std_val > 0:
                        df_final[col] = (df_final[col] - mean_val) / std_val
    
    # Fallback for missing narratives (impute from DMA or national average)
    if 'n_buy' not in df_final.columns or df_final['n_buy'].isna().mean() > 0.5:
        print("  High missing rate for narratives. Imputing from DMA/national averages...")
        if 'n_buy' not in df_final.columns:
            df_final['n_buy'] = 0
            df_final['n_risk'] = 0
        
        # Impute: first try DMA-quarter average, then national quarter average
        quarter_avg = df_final.groupby('quarter')[['n_buy', 'n_risk']].transform('mean')
        df_final['n_buy'] = df_final['n_buy'].fillna(quarter_avg['n_buy'])
        df_final['n_risk'] = df_final['n_risk'].fillna(quarter_avg['n_risk'])
    
    # Create lagged variables
    df_final['n_buy_lag'] = df_final.groupby('region')['n_buy'].shift(1)
    df_final['n_risk_lag'] = df_final.groupby('region')['n_risk'].shift(1)
    df_final['volume_growth_lag'] = df_final.groupby('region')['volume_growth'].shift(1)
    
    # Inventory log and lag
    df_final['ln_inventory'] = np.log(df_final['inventory'].clip(lower=1))
    df_final['ln_inventory_lag'] = df_final.groupby('region')['ln_inventory'].shift(1)
    
    # Jumbo Exposure Proxy (price rank)
    metro_price = df_final.groupby('region')['median_sale_price'].mean()
    metro_price_rank = metro_price.rank(pct=True)
    df_final['jumbo_exposure'] = df_final['region'].map(metro_price_rank)
    
    # Post Policy Indicator
    df_final['post_policy'] = (df_final['quarter'] >= '2014-01-01').astype(int)
    
    # Panel Verification
    n_metros = df_final['region'].nunique()
    n_quarters = df_final['quarter'].nunique()
    print(f"\n  PANEL CHECK:")
    print(f"    Metros: {n_metros}")
    print(f"    Quarters: {n_quarters}")
    print(f"    Theoretical max: {n_metros * n_quarters}")
    print(f"    Actual obs: {len(df_final)}")
    
    # Save
    out_path = os.path.join(PROCESSED_DIR, "panel_data_real.csv")
    if os.path.exists(out_path):
        os.remove(out_path)
    df_final.to_csv(out_path, index=False)
    print(f"\n✓ Saved to {out_path}")

if __name__ == "__main__":
    main()
