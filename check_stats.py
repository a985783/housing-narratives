
import sys
import os
import pandas as pd
# Add current directory to path
sys.path.append(os.getcwd())

from code.04_analysis_real import load_and_clean_data

if __name__ == "__main__":
    df = load_and_clean_data()
    print('\n' + '='*30)
    print("PHASE 1.6 VERIFICATION STATS")
    print('='*30)
    print(f"Final Regression N: {len(df)}")
    n_metros = df.index.get_level_values('region').nunique()
    n_dmas = df['dma_code'].nunique()
    print(f"Unique Metros (Entities): {n_metros}")
    print(f"Unique DMAs (Clusters):   {n_dmas}")
    print('='*30)
    
    # List top 5 DMAs by metro count
    print("\nTop 5 DMAs by Metro Count:")
    print(df.reset_index().drop_duplicates('region')['dma_code'].value_counts().head(5))
