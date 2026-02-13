
import pandas as pd
import os

# 1. Define the Dictionary EXACTLY as in 04_analysis_real.py (Step 710)
METRO_DMA_CROSSWALK = {
    # 1. Northeast
    'New York': 'US-NY-501', 'Newark': 'US-NY-501', 'Jersey City': 'US-NY-501',
    'Nassau': 'US-NY-501', 'Bridgeport': 'US-NY-501', 'Stamford': 'US-NY-501',
    'Philadelphia': 'US-PA-504', 'Camden': 'US-PA-504', 'Wilmington': 'US-PA-504', 'Allentown': 'US-PA-504',
    'Boston': 'US-MA-506', 'Cambridge': 'US-MA-506', 'Newton': 'US-MA-506', 'Worcester': 'US-MA-506',
    'Providence': 'US-RI-521', 'Hartford': 'US-CT-533', 'New Haven': 'US-CT-533',
    'Pittsburgh': 'US-PA-508', 'Buffalo': 'US-NY-514', 'Rochester': 'US-NY-538', 'Albany': 'US-NY-532',
    # 2. Midwest
    'Chicago': 'US-IL-602', 'Naperville': 'US-IL-602', 'Elgin': 'US-IL-602',
    'Detroit': 'US-MI-505', 'Warren': 'US-MI-505', 'Ann Arbor': 'US-MI-505',
    'Minneapolis': 'US-MN-613', 'St. Paul': 'US-MN-613',
    'Cleveland': 'US-OH-510', 'Columbus': 'US-OH-535', 'Cincinnati': 'US-OH-515',
    'Indianapolis': 'US-IN-527', 'St. Louis': 'US-MO-609', 'Kansas City': 'US-MO-616',
    'Milwaukee': 'US-WI-617', 'Grand Rapids': 'US-MI-563', 'Oklahoma City': 'US-OK-650',
    'Tulsa': 'US-OK-671', 'Omaha': 'US-NE-740', 'Des Moines': 'US-IA-679',
    # 3. South
    'Washington': 'US-DC-511', 'Arlington': 'US-DC-511', 'Alexandria': 'US-DC-511', 'Baltimore': 'US-MD-512',
    'Atlanta': 'US-GA-524', 'Sandy Springs': 'US-GA-524',
    'Miami': 'US-FL-528', 'Fort Lauderdale': 'US-FL-528', 'West Palm Beach': 'US-FL-548',
    'Tampa': 'US-FL-539', 'St. Petersburg': 'US-FL-539', 'Orlando': 'US-FL-534', 'Jacksonville': 'US-FL-561',
    'Charlotte': 'US-NC-517', 'Raleigh': 'US-NC-560', 'Durham': 'US-NC-560',
    'Nashville': 'US-TN-659', 'Memphis': 'US-TN-640', 'New Orleans': 'US-LA-622',
    'Louisville': 'US-KY-529', 'Birmingham': 'US-AL-630', 'Richmond': 'US-VA-556',
    'Virginia Beach': 'US-VA-544', 'Norfolk': 'US-VA-544', 'Greensboro': 'US-NC-518',
    'Knoxville': 'US-TN-557', 'Greenville': 'US-SC-567', 'Columbia': 'US-SC-546', 'Charleston': 'US-SC-519',
    # 4. Texas / Southwest
    'Dallas': 'US-TX-623', 'Fort Worth': 'US-TX-623', 'Arlington': 'US-TX-623', 'Plano': 'US-TX-623',
    'Houston': 'US-TX-618', 'The Woodlands': 'US-TX-618',
    'San Antonio': 'US-TX-641', 'Austin': 'US-TX-635', 'El Paso': 'US-TX-765', 'McAllen': 'US-TX-636',
    'Phoenix': 'US-AZ-753', 'Mesa': 'US-AZ-753', 'Scottsdale': 'US-AZ-753', 'Tucson': 'US-AZ-789',
    'Albuquerque': 'US-NM-790', 'Las Vegas': 'US-NV-839',
    # 5. West
    'Los Angeles': 'US-CA-803', 'Long Beach': 'US-CA-803', 'Anaheim': 'US-CA-803', 'Riverside': 'US-CA-803',
    'San Francisco': 'US-CA-807', 'Oakland': 'US-CA-807', 'Berkeley': 'US-CA-807', 'San Jose': 'US-CA-807',
    'Sunnyvale': 'US-CA-807', 'Santa Clara': 'US-CA-807',
    'San Diego': 'US-CA-825', 'Sacramento': 'US-CA-862', 'Fresno': 'US-CA-866', 'Bakersfield': 'US-CA-800',
    'Seattle': 'US-WA-819', 'Tacoma': 'US-WA-819', 'Bellevue': 'US-WA-819',
    'Portland': 'US-OR-820', 'Vancouver': 'US-OR-820',
    'Denver': 'US-CO-751', 'Aurora': 'US-CO-751', 'Salt Lake City': 'US-UT-770',
    'Honolulu': 'US-HI-744', 'Boise': 'US-ID-757', 'Spokane': 'US-WA-881',
}

def get_dma_for_metro(metro_name):
    for pattern, dma_code in METRO_DMA_CROSSWALK.items():
        if pattern.lower() in metro_name.lower():
            return dma_code
    return None

DATA_PATH = "data/processed/panel_data_real.csv"
OUTPUT_CSV = "data/mappings/metro_dma_crosswalk_main_208.csv"

def main():
    print("Generating MAIN Crosswalk for Replication (208 Metros)...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")
        
    df = pd.read_csv(DATA_PATH)
    
    # Apply Mapping
    df['dma_code'] = df['region'].apply(get_dma_for_metro)
    
    # Filter for Complete Case Logic (same as 04)
    # We only care about Metros that end up in the sample
    # But wait, 04 drops invalid n_buy too?
    # Yes, we want the crosswalk for the FINAL sample.
    
    # Distinct mapping
    df_cw = df[['region', 'dma_code']].drop_duplicates()
    
    # Filter out None/NaN
    df_cw = df_cw.dropna(subset=['dma_code'])
    
    print(f"Mapped {len(df_cw)} unique Metros.")
    
    # Rename for export
    df_cw = df_cw.rename(columns={'region': 'Metro', 'dma_code': 'DMA_Code'})
    
    # Apply Manual Fixes if needed? (No, we export what works)
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_cw.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df_cw)} mappings to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
