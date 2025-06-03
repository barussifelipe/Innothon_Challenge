import numpy as np
import pandas as pd 
import os 

# --- Configuration ---
# Define the path to your raw consumption data
RAW_CONSUMPTION_FILE = "/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data/Consumption.csv"
# Define the output directory for pivoted data
OUTPUT_PIVOTED_DATA_DIR = "/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Study_datasets/Consumption/new_pivoted_data"

# List of known problematic supplies to drop.
# Add IDs here based on the output of your validation script (e.g., constant consumption, very large gaps).
PROBLEMATIC_SUPPLY_IDS = ['SUPPLY062', 'SUPPLY071'] 
# Example: If SUPPLY062 has zero variance, and SUPPLY018, SUPPLY092, SUPPLY071 have large gaps.
# Add any other supplies identified by your validation script as unsuitable.

# Minimum number of days a supply must have to be included (optional filter).
# Supplies with fewer days than this will be skipped. Set to 0 to disable this filter.
MIN_DAYS_LOGGED = 1 # Example: Require at least 1 year of data

print(f"--- Starting Data Preprocessing ---")
print(f"Raw data file: {RAW_CONSUMPTION_FILE}")

# --- 1. Load Raw Quarter Hour Consumption Data ---
try:
    consumption_data = pd.read_csv(
        RAW_CONSUMPTION_FILE,
        encoding="utf-16",
        sep="\t",
        decimal=',', # Crucial for handling comma decimals in the raw data
        header=0
    )
    print(f"Loaded raw data: {len(consumption_data)} entries.")
    print("Raw data dtypes:\n", consumption_data.dtypes)
except FileNotFoundError:
    print(f"Error: Raw consumption file not found at {RAW_CONSUMPTION_FILE}. Exiting.")
    exit()
except Exception as e:
    print(f"Error loading raw consumption data: {e}. Exiting.")
    exit()

# --- NEW CHECK: Validate 'id' column range in raw data ---
# This checks if all quarter-hour IDs are within the expected 1 to 96 range.
print("\n--- Validating 'id' column range in raw data ---")
if not consumption_data['id'].between(1, 96).all():
    invalid_ids = consumption_data[~consumption_data['id'].between(1, 96)]['id'].unique()
    print(f"  Warning: 'id' column contains values outside the expected 1-96 range. Invalid IDs: {invalid_ids}")
    # Action: Filter out rows with invalid 'id's. These rows cannot be pivoted correctly.
    consumption_data = consumption_data[consumption_data['id'].between(1, 96)].copy()
    print(f"  Removed rows with invalid 'id' values. Remaining entries: {len(consumption_data)}")
else:
    print("  'id' column values are all within the expected 1-96 range.")


# --- 2. Combine Date Columns into YYYY/MM/DD Format ---
print("\n--- Formatting date column ---")
consumption_data["meas_ym"] = consumption_data['meas_ym'].astype(str).str[:4] + '/' + \
                              consumption_data['meas_ym'].astype(str).str[4:6]
consumption_data['meas_dd'] = consumption_data['meas_dd'].astype(str).str.zfill(2)
consumption_data['date'] = consumption_data['meas_ym'] + '/' + consumption_data['meas_dd']

# Drop the original meas_ym and meas_dd columns
consumption_data.drop(columns=['meas_ym', 'meas_dd'], inplace=True)
print("Date column created and original date columns dropped.")

# --- 3. Drop Known Problematic Supplies (If defined) ---
if PROBLEMATIC_SUPPLY_IDS:
    initial_supplies_count = consumption_data['Supply_ID'].nunique()
    consumption_data = consumption_data[~consumption_data['Supply_ID'].isin(PROBLEMATIC_SUPPLY_IDS)].copy()
    final_supplies_count = consumption_data['Supply_ID'].nunique()
    print(f"\n--- Dropped problematic supplies based on PROBLEMATIC_SUPPLY_IDS ---")
    print(f"Removed {initial_supplies_count - final_supplies_count} supplies.")
    print(f"Remaining unique supplies: {final_supplies_count}")

# --- 4. Process Each Supply ID Separately ---
print("\n--- Processing each supply ID ---")
supply_ids = consumption_data['Supply_ID'].unique()
supply_dataframes = {}

for supply_id in supply_ids:
    supply_data = consumption_data[consumption_data['Supply_ID'] == supply_id].copy()
    # Drop 'Supply_ID' as it's no longer needed in the individual dataframe
    # Drop 'magnitude' as it's not used in the model
    supply_data.drop(columns=['Supply_ID', 'magnitude'], inplace=True) 
    
    # Convert 'val' to numeric. 'errors='coerce' will turn any non-numeric into NaN.
    # This is a safeguard, as decimal=',' in read_csv should handle most cases.
    supply_data['val'] = pd.to_numeric(supply_data['val'], errors='coerce')
    
    # IMPORTANT: No NaN imputation (interpolate/fillna) here, as per your choice.
    # NaNs introduced by pivoting (for missing quarter-hours) or from errors='coerce'
    # will be handled by data.dropna() in the DataLoader.

    supply_dataframes[supply_id] = supply_data

print(f"Created {len(supply_dataframes)} dataframes for individual supplies.")

# --- 5. Pivot Data for Each Supply ---
print("\n--- Pivoting data (date as row, quarter-hours as columns) ---")
final_pivoted_dataframes = {}
for supply_id, df in supply_dataframes.items():
    # Ensure all 96 quarter-hour IDs are expected columns
    all_ids = np.arange(1, 97) 
    
    # Pivot the data: 'date' becomes index, 'id' becomes columns, 'val' fills values.
    df_pivoted = df.pivot(index='date', columns='id', values='val')
    
    # Ensure all 96 columns are present. If a quarter-hour ID was missing for a day,
    # pivot would leave it out. We add it back with NaN.
    missing_cols = set(all_ids) - set(df_pivoted.columns)
    for col in missing_cols:
        df_pivoted[col] = np.nan # Use np.nan, DataLoader's dropna() will handle it

    # Sort columns numerically (1, 2, ..., 96) for consistent model input order.
    df_pivoted = df_pivoted.reindex(columns=sorted(df_pivoted.columns))

    final_pivoted_dataframes[supply_id] = df_pivoted

print("Pivoting complete for all supplies.")

# --- 6. Filter by Minimum Days Logged (Optional) ---
if MIN_DAYS_LOGGED > 0:
    print(f"\n--- Filtering supplies with less than {MIN_DAYS_LOGGED} days logged ---")
    initial_filtered_count = len(final_pivoted_dataframes)
    supplies_to_keep = {}
    for supply_id, df in final_pivoted_dataframes.items():
        if len(df) >= MIN_DAYS_LOGGED:
            supplies_to_keep[supply_id] = df
        else:
            print(f"  Skipping {supply_id}: Only {len(df)} days logged (less than {MIN_DAYS_LOGGED}).")
    final_pivoted_dataframes = supplies_to_keep
    print(f"Filtered out {initial_filtered_count - len(final_pivoted_dataframes)} supplies.")
    print(f"Remaining supplies after filtering: {len(final_pivoted_dataframes)}")


# --- 7. Save Pivoted Data to Files ---
print(f"\n--- Saving pivoted data to {OUTPUT_PIVOTED_DATA_DIR} ---")
if not os.path.exists(OUTPUT_PIVOTED_DATA_DIR):
    os.makedirs(OUTPUT_PIVOTED_DATA_DIR)

for supply_id, df in final_pivoted_dataframes.items():
    file_path = os.path.join(OUTPUT_PIVOTED_DATA_DIR, f"{supply_id}.csv")
    # Save with date as index. pd.to_csv typically uses '.' for decimals,
    # but DataLoader and validation script use decimal=',' to read it.
    df.to_csv(file_path, index=True) 
    print(f"Saved {supply_id} data to {file_path}")

print("\n--- Data Preprocessing Complete ---")