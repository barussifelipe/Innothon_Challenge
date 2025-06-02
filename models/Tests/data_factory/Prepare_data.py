import pandas as pd
import os

# Paths
PROVIDED_DATA_PATH = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data'
DATA_OUTPUT_PATH = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Tests/Data'
CONSUMPTION_FILE = 'Consumption.csv'
LABELS_FILE = 'Labels.csv'

# Fixed number of days per supply
fixed_days = 10
FINAL_DF_FILE = str(fixed_days) + '_days.csv'

# Helper function
def id_to_timedelta(id_val):
    """
    Converts quarter-hour ID (1-96) to a pandas Timedelta object.
    id=1 -> 00:00, id=96 -> 23:45
    """
    minutes = (id_val - 1) * 15
    return pd.Timedelta(minutes=minutes)

def get_complete_day_metadata(df_with_timestamps: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates metadata for each day of each supply, identifying 'complete' days.
    A 'complete day' is defined as having exactly 96 unique quarter-hour records
    AND no NaN values in the 'val' column for that day.

    Returns a DataFrame with 'Supply_ID', 'Date', 'is_complete_day', 'record_count', 'val_null_count'.
    """
    # Extract just the date part from the Timestamp
    df_with_timestamps['Date_Only'] = df_with_timestamps['Timestamp'].dt.normalize()

    # Group by Supply_ID and Date_Only to analyze each day
    daily_stats = df_with_timestamps.groupby(['Supply_ID', 'Date_Only']).agg(
        record_count=('id', 'nunique'), # Count unique IDs for each day
        val_null_count=('val', lambda x: x.isnull().sum()) # Count NaN 'val' entries
    ).reset_index()

    # Define a complete day: 96 unique records AND no NaN 'val' entries
    daily_stats['is_complete_day'] = (daily_stats['record_count'] == 96) & \
                                     (daily_stats['val_null_count'] == 0)

    # Clean up temporary column
    df_with_timestamps.drop(columns=['Date_Only'], inplace=True)

    return daily_stats[['Supply_ID', 'Date_Only', 'is_complete_day', 'record_count', 'val_null_count']]


def filter_and_truncate_by_complete_days(df_consumption_raw: pd.DataFrame, num_required_days: int) -> pd.DataFrame:
    """
    Filters supplies based on a minimum number of complete days and then
    truncates them to keep only the first 'num_required_days' complete days.

    Args:
        df_consumption_raw (pd.DataFrame): The DataFrame with 'Supply_ID', 'Timestamp', 'val', 'id',
                                           and other relevant metadata.
        num_required_days (int): The minimum number of complete days a supply must have,
                                 and the number of complete days to truncate to.

    Returns:
        pd.DataFrame: A new DataFrame containing only the selected and truncated data,
                      where each supply has exactly num_required_days * 96 entries,
                      and all 'val' entries are non-null.
    """
    print(f"\n--- Stage 1: Identifying complete days and filtering supplies ---")
    # Get metadata for each day of each supply
    daily_metadata = get_complete_day_metadata(df_consumption_raw.copy()) # Pass a copy to avoid modifying original

    # Count the total number of complete days for each supply
    supply_total_complete_days = daily_metadata[daily_metadata['is_complete_day']] \
                                .groupby('Supply_ID')['Date_Only'].count()

    # Identify supplies that meet the minimum requirement (at least num_required_days complete days)
    qualified_supplies = supply_total_complete_days[
        supply_total_complete_days >= num_required_days
    ].index.tolist()

    print(f"Initial unique supplies: {df_consumption_raw['Supply_ID'].nunique()}")
    print(f"Supplies with at least {num_required_days} complete days: {len(qualified_supplies)}")

    if not qualified_supplies:
        print("No supplies meet the minimum complete day requirement. Returning empty DataFrame.")
        return pd.DataFrame(columns=df_consumption_raw.columns) # Return an empty DataFrame with original columns

    # Filter the original DataFrame to keep only qualified supplies
    df_filtered_supplies = df_consumption_raw[df_consumption_raw['Supply_ID'].isin(qualified_supplies)].copy()
    
    print(f"\n--- Stage 2: Truncating to the first {num_required_days} complete days ---")
    final_data_list = []
    
    # Iterate through each qualified supply
    for supply_id in qualified_supplies:
        supply_data = df_filtered_supplies[df_filtered_supplies['Supply_ID'] == supply_id].copy()
        
        # Get the complete dates for this specific supply, sorted chronologically
        complete_dates_for_supply = daily_metadata[
            (daily_metadata['Supply_ID'] == supply_id) & 
            (daily_metadata['is_complete_day'])
        ].sort_values(by='Date_Only')['Date_Only'].tolist()

        # Select only the first num_required_days complete dates
        selected_dates = complete_dates_for_supply[:num_required_days]

        # Filter the supply's data to include only records from these selected complete dates
        # Note: We convert Timestamp to Date_Only for direct comparison with selected_dates
        supply_data['Date_Only'] = supply_data['Timestamp'].dt.normalize()
        truncated_supply_data = supply_data[supply_data['Date_Only'].isin(selected_dates)].copy()
        
        # Drop the temporary Date_Only column
        truncated_supply_data.drop(columns=['Date_Only'], inplace=True)

        # Ensure that each truncated supply has exactly num_required_days * 96 entries
        # This is a final check, it should be true if 'is_complete_day' and filtering logic are correct
        expected_rows = num_required_days * 96
        if len(truncated_supply_data) != expected_rows:
            print(f"Warning: {supply_id} expected {expected_rows} rows but got {len(truncated_supply_data)}. This indicates a data anomaly.")
            # For robustness, you might choose to drop this supply or log it for manual review
            # For now, we proceed but flag it.
            
        final_data_list.append(truncated_supply_data)

    if not final_data_list:
        print("No supplies remained after truncation. Returning empty DataFrame.")
        return pd.DataFrame(columns=df_consumption_raw.columns)

    return pd.concat(final_data_list, ignore_index=True)


# --- Main Data Loading and Initial Processing ---
print('Loading in datasets...\n')
df_consumption = pd.read_csv(os.path.join(PROVIDED_DATA_PATH, CONSUMPTION_FILE), encoding='utf-16', sep='\t', decimal=',')
df_labels = pd.read_csv(os.path.join(PROVIDED_DATA_PATH, LABELS_FILE), encoding='utf-16', sep='\t', decimal=',')

# Drop known 'problematic' supplies (retain this filter as it's an initial data cleaning step)
extra_supplies_for_smaller_dataset = ['SUPPPLY090','SUPPLY091','SUPPLY093', 'SUPPLY095', 'SUPPLY096', 'SUPPLY097', 'SUPPLY098', 'SUPPLY099', 'SUPPLY100']
NaN_Supplies_drop = ['SUPPLY018','SUPPLY019', 'SUPPLY082', 'SUPPLY094', 'SUPPLY077', 'SUPPLY085']
not_enough_days_drop = ['SUPPLY001', 'SUPPLY069', 'SUPPLY071', 'SUPPLY092'] # These supplies might already be dropped by new logic
final_drop = NaN_Supplies_drop + not_enough_days_drop + extra_supplies_for_smaller_dataset
df_consumption = df_consumption[~df_consumption['Supply_ID'].isin(final_drop)]
unique_supplies_initial = df_consumption['Supply_ID'].unique()
print(f'Dropped manually specified problematic supplies: {NaN_Supplies_drop + not_enough_days_drop}\n')
print(f'Supplies count after initial manual drop: {len(unique_supplies_initial)}')

# Merge labels with consumption data
df_consumption = pd.merge(df_consumption, df_labels, on='Supply_ID', how='left')

# Create Proper Timestamps
df_consumption['meas_ym_str'] = df_consumption['meas_ym'].astype(str)
df_consumption['date_str'] = df_consumption['meas_ym_str'].str[:4] + '-' + \
                            df_consumption['meas_ym_str'].str[4:6] + '-' + \
                            df_consumption['meas_dd'].astype(str).str.zfill(2)

df_consumption['Date'] = pd.to_datetime(df_consumption['date_str'], errors='coerce')
df_consumption['id'] = df_consumption['id'].astype(int) # Ensure 'id' is int for nunique counting
df_consumption['Timestamp'] = df_consumption['Date'] + df_consumption['id'].apply(id_to_timedelta)

# Drop intermediate and unneeded columns
df_consumption.drop(columns=['meas_ym_str', 'date_str', 'Date', 'meas_ym', 'meas_dd', 'magnitude'], inplace=True) # Keep 'id' for nunique check

# Handle rows where Timestamp could not be parsed (if any)
print(f"Number of NaN values in 'Timestamp' before dropping: {df_consumption['Timestamp'].isnull().sum()}")
df_consumption.dropna(subset=['Timestamp'], inplace=True)
print(f"Number of NaN values in 'Timestamp' after dropping: {df_consumption['Timestamp'].isnull().sum()}")

# Sort by Supply_ID and Timestamp for correct chronological processing
df_consumption = df_consumption.sort_values(by=['Supply_ID', 'Timestamp']).reset_index(drop=True)

# Check for NaN in 'val' (original NaNs that were present before this new logic)
print(f"Number of original NaN values in 'val' before strict filtering: {df_consumption['val'].isnull().sum()}")

# Define Is_Non_Regular (do this before dropping CLUSTER if you need it later)
df_consumption['Is_Non_Regular'] = df_consumption['CLUSTER'].apply(
    lambda x: 1 if x in ['Frode', 'Anomalia'] else 0
)
# df_consumption.drop(columns=['CLUSTER'], inplace=True) # You can drop CLUSTER here if no longer needed

# --- Apply the new strict filtering and truncation logic ---
df_consumption_final = filter_and_truncate_by_complete_days(df_consumption, num_required_days=fixed_days)

print(f'\nFinal dataset processing complete.')

if not df_consumption_final.empty:
    print(f'Total unique supplies in final dataset: {df_consumption_final["Supply_ID"].nunique()}')
    print(f'Final dataset head\n{df_consumption_final.head()}')
    print(f'\nFinal dataset tail\n{df_consumption_final.tail()}')
    print(f"Number of NaN values in 'val' in final dataset: {df_consumption_final['val'].isnull().sum()}")
    print(f"Total entries in final dataset: {len(df_consumption_final)}")

    # Verify a sample supply for consistency (e.g., if SUPPLY023 was kept)
    sample_supply_id_to_check = 'SUPPLY023' # Or choose one from qualified_supplies
    if sample_supply_id_to_check in df_consumption_final['Supply_ID'].unique():
        check_df = df_consumption_final[df_consumption_final['Supply_ID'] == sample_supply_id_to_check]
        print(f"\nVerification for {sample_supply_id_to_check}:")
        print(f"Number of entries: {len(check_df)}")
        print(f"Expected entries for {fixed_days} days: {fixed_days * 96}")
        print(f"Number of NaN values in 'val' for {sample_supply_id_to_check}: {check_df['val'].isnull().sum()}")
        print(f"First 5 timestamps:\n{check_df['Timestamp'].head()}")
        print(f"Last 5 timestamps:\n{check_df['Timestamp'].tail()}")
    else:
        print(f"\n{sample_supply_id_to_check} was not found in the final dataset (likely did not meet {fixed_days} complete days).")

    # Save the final DataFrame
    df_consumption_final.to_csv(os.path.join(DATA_OUTPUT_PATH, FINAL_DF_FILE), index=False)
    print(f'Saved final dataset to {os.path.join(DATA_OUTPUT_PATH, FINAL_DF_FILE)}')
else:
    print("Final DataFrame is empty as no supplies met the stringent criteria.")