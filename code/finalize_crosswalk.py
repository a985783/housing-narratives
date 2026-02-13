
import pandas as pd
import os

# 1. Define the Dictionary (Source of Truth from Phase 1.5)
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

CSV_PATH = "data/mappings/metro_dma_crosswalk_v20260201.csv"
DATA_PATH = "data/processed/panel_data_real.csv"

def get_dma_dict(metro_name):
    for pattern, dma_code in METRO_DMA_CROSSWALK.items():
        if pattern.lower() in metro_name.lower():
            return dma_code
    return None

def get_dma_csv(metro_name, cw_list):
    for pattern, dma_code in cw_list:
        if pattern.lower() in metro_name.lower():
            return dma_code
    return None

def main():
    print("Generating Verified Crosswalk CSV...")
    
    # 1. Write Dictionary to CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df_cw = pd.DataFrame(list(METRO_DMA_CROSSWALK.items()), columns=['Metro_Pattern', 'DMA_Code'])
    df_cw.to_csv(CSV_PATH, index=False)
    print(f"Saved to {CSV_PATH}")
    
    # 2. Verify Mapping Consistency
    print("Verifying against real data...")
    if not os.path.exists(DATA_PATH):
        print("Warning: Data path not found, can't verify against real panel.")
        return

    df = pd.read_csv(DATA_PATH)
    unique_metros = df['region'].unique()
    
    print(f"Total Unique Regions in Data: {len(unique_metros)}")
    
    # Map using Dict
    mapped_dict = {m: get_dma_dict(m) for m in unique_metros}
    count_dict = sum(1 for v in mapped_dict.values() if v is not None)
    
    # Map using CSV (Simulate Load)
    df_loaded = pd.read_csv(CSV_PATH)
    cw_list = list(zip(df_loaded['Metro_Pattern'], df_loaded['DMA_Code']))
    mapped_csv = {m: get_dma_csv(m, cw_list) for m in unique_metros}
    count_csv = sum(1 for v in mapped_csv.values() if v is not None)
    
    print(f"Matches using Dict: {count_dict}")
    print(f"Matches using CSV:  {count_csv}")
    
    if count_dict == count_csv:
        print("SUCCESS: CSV Mapping matches Dictionary Mapping.")
        
        # Print unmatched stats
        unmatched = [m for m, v in mapped_csv.items() if v is None]
        print(f"Unmatched Metros: {len(unmatched)}")
        print("Sample Unmatched:")
        for m in unmatched[:20]:
            print(f"  {m}")
        # Save unmatched
        with open("data/mappings/unmatched_metros.txt", "w") as f:
            for m in unmatched:
                f.write(f"{m}\n")
    else:
        print("FAILURE: Mismatch in mapping counts!")

if __name__ == "__main__":
    main()
