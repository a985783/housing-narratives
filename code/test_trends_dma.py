
from pytrends.request import TrendReq
import pandas as pd

def test_dma_resolution():
    print("Testing DMA resolution for interest_by_region...")
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        kw_list = ["buy a house"]
        pytrends.build_payload(kw_list, timeframe='2023-01-01 2023-12-31', geo='US')
        
        # Test 1: DMA Resolution
        print("1. Requesting interest_by_region(resolution='DMA')...")
        df_dma = pytrends.interest_by_region(resolution='DMA', inc_low_vol=True, inc_geo_code=True)
        print(df_dma.head())
        print(f"Rows: {len(df_dma)}")
        
        # Check if we have DMA codes
        if not df_dma.empty:
            print("Success! Found DMA data.")
            # Verify if index is DMA code or Name
            print(f"Index sample: {df_dma.index[:5].tolist()}")
        else:
            print("Returned empty dataframe.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_dma_resolution()
