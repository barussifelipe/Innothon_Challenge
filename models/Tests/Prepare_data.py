import pandas as pd
import os

#Paths 
PROVIDED_DATA_PATH = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data'
DATA_OUTPUT_PATH = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Tests/Data'
CONSUMPTION_FILE = 'Consumption.csv'
LABELS_FILE = 'Labels.csv'

#Fixed number of days per supply
fixed_days = 1300
FINAL_DF_FILE = str(fixed_days) + '_day.csv'

# Helper function
def id_to_timedelta(id_val):
    """
    Converts quarter-hour ID (1-96) to a pandas Timedelta object.
    id=1 -> 00:00, id=96 -> 23:45
    """
    minutes = (id_val - 1) * 15
    return pd.Timedelta(minutes=minutes)

def enforce_fixed_days(group, num_days, supply_id_val):
    """
    Truncates or pads a group of data to have a fixed number of days.
    If the group has more than num_days, it truncates to the first num_days.
    If the group has fewer than num_days, it pads with NaN values to reach num_days.
    """

    print(f"Enforcing days on {supply_id_val}")

    start_date = group['Timestamp'].min().normalize()
    # The inclusive end date of the data period, not just the filter
    period_end_date_inclusive = start_date + pd.Timedelta(days=num_days - 1)
    
    # Generate all expected timestamps for the *exact* days in the range, including leap days
    # This is the most accurate way to get the true expected count
    full_expected_range = pd.date_range(start=start_date, end=period_end_date_inclusive.replace(hour=23, minute=45), freq='15min')
    expected_num_timestamps = len(full_expected_range)

    # Filter the group to include only data within the actual range
    # It's crucial that this filter aligns with the period used for expected_num_timestamps
    # We want data from `start_date` up to the *last minute* of the `num_days` period.
    filter_end_ts = start_date + pd.Timedelta(days=num_days) - pd.Timedelta(minutes=15)
    group = group[(group['Timestamp'] >= start_date) & (group['Timestamp'] <= filter_end_ts)].copy()

    num_missing = expected_num_timestamps - len(group)
    # Pad the group with NaN values if necessary
    if num_missing > 0:
        print(f'Missing timestamps found in\n{group.tail()}')
        print(f'Expected: {expected_num_timestamps}, Actual (before padding): {len(group)}, Missing: {num_missing}')
        
        # Create a DataFrame with the missing timestamps, continuing from the last existing timestamp
        # This will only append at the end, not fill internal gaps, as per original logic's intent.
        last_timestamp = group['Timestamp'].max()
        # If group is empty (e.g., after filtering), last_timestamp will be NaT. Handle this.
        if pd.isna(last_timestamp):
            # If the group is empty, start padding from the 'start_date'
            padding_start_ts = start_date
        else:
            padding_start_ts = last_timestamp

        new_timestamps = [padding_start_ts + pd.Timedelta(minutes=15 * i) for i in range(1, num_missing + 1)]
        padding = pd.DataFrame({'Timestamp': new_timestamps})

        # To retain Supply_ID, CLUSTER, and Is_Non_Regular for padded rows, fill them from the group.
        # This assumes these values are consistent for a given Supply_ID.
        padding['Supply_ID'] = supply_id_val
        padding['CLUSTER'] = group['CLUSTER'].iloc[0] if not group.empty else None
        padding['Is_Non_Regular'] = group['Is_Non_Regular'].iloc[0] if not group.empty else None

        # Merge the padding DataFrame with the original group
        group = pd.concat([group, padding], ignore_index=True)

    # Ensure Supply_ID is present in all rows of the group before returning
    group['Supply_ID'] = supply_id_val

    # Sort and then truncate to exactly 'expected_num_timestamps'
    # This handles both excess data and ensures the final length is 'expected_num_timestamps'.
    return group.sort_values(by='Timestamp').head(expected_num_timestamps)

#Load in raw datasets
print('Loading in datasets...\n')
df_consumption = pd.read_csv(os.path.join(PROVIDED_DATA_PATH, CONSUMPTION_FILE), encoding='utf-16', sep='\t', decimal=',')
df_labels = pd.read_csv(os.path.join(PROVIDED_DATA_PATH, LABELS_FILE), encoding='utf-16', sep='\t', decimal=',')


#Drop known 'problematic' supplies
NaN_Supplies_drop = ['SUPPLY018','SUPPLY019', 'SUPPLY082', 'SUPPLY094', 'SUPPLY077', 'SUPPLY085']
not_enough_days_drop = ['SUPPLY001', 'SUPPLY069', 'SUPPLY071', 'SUPPLY092']
df_consumption = df_consumption[~df_consumption['Supply_ID'].isin(NaN_Supplies_drop + not_enough_days_drop)]
unique_supplies = df_consumption['Supply_ID'].unique()
print(f'Dropped problematic supplies {NaN_Supplies_drop + not_enough_days_drop}\n')
print(f'Supplies count: {len(unique_supplies)}')

# Merge labels to count regular and non-regular supplies
df_labels_filtered = df_labels[df_labels['Supply_ID'].isin(unique_supplies)]
regular_supplies_count = len(df_labels_filtered[df_labels_filtered['CLUSTER'] == 'Regolare'])
non_regular_supplies_count = len(df_labels_filtered[df_labels_filtered['CLUSTER'].isin(['Frode', 'Anomalia'])])

print(f'Regular supplies count: {regular_supplies_count}')
print(f'Non-Regular supplies count: {non_regular_supplies_count}')
print(f'Datasets loaded\n{df_consumption.head()}')

# --- 2. Merge Labels with Consumption Data ---
df_consumption = pd.merge(df_consumption, df_labels, on='Supply_ID', how='left')

# --- 3. Create Proper Timestamps ---
df_consumption['meas_ym_str'] = df_consumption['meas_ym'].astype(str)
df_consumption['date_str'] = df_consumption['meas_ym_str'].str[:4] + '-' + \
                            df_consumption['meas_ym_str'].str[4:6] + '-' + \
                            df_consumption['meas_dd'].astype(str).str.zfill(2) # Ensure day is 2 digits

df_consumption['Date'] = pd.to_datetime(df_consumption['date_str'], errors='coerce')
df_consumption['id'] = df_consumption['id'].astype(int)
df_consumption['Timestamp'] = df_consumption['Date'] + df_consumption['id'].apply(id_to_timedelta)

# Drop intermediate and unneeded columns
df_consumption = df_consumption.drop(columns=['meas_ym_str', 'date_str', 'Date', 'meas_ym', 'meas_dd', 'id', 'magnitude'])

# Check for NaN values in 'Timestamp' before dropping
print(f"Number of NaN values in 'Timestamp': {df_consumption['Timestamp'].isnull().sum()}")
# # Drop rows where Timestamp could not be parsed (if any)
# df_consumption.dropna(subset=['Timestamp'], inplace=True)

# Sort by Supply_ID and Timestamp for correct processing
df_consumption = df_consumption.sort_values(by=['Supply_ID', 'Timestamp']).reset_index(drop=True)

# Check for NaN in values in 'val'
print(f"Number of NaN values in 'val': {df_consumption['val'].isnull().sum()}")

# Define Is_Non_Regular
df_consumption['Is_Non_Regular'] = df_consumption['CLUSTER'].apply(
    lambda x: 1 if x in ['Frode', 'Anomalia'] else 0
)

df_consumption.drop(columns=['CLUSTER'], inplace=True)

# Enforce a fixed number of days per supply 
print(f'Enforcing {fixed_days} on each supply')

df_consumption = df_consumption.groupby('Supply_ID', group_keys=False).apply(
    lambda group: enforce_fixed_days(group, num_days=fixed_days, supply_id_val=group.name)
)
print(f'Days fixed: {fixed_days}')

print(f'Final dataset head\n{df_consumption.head()}')

df_consumption.to_csv(os.path.join(DATA_OUTPUT_PATH, FINAL_DF_FILE))
