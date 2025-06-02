import pandas as pd
import numpy as np

df_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data/Consumption.csv'
labels_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data/Labels.csv'


# --- Start by loading your actual raw data ---
# Replace 'your_raw_data.csv' with the actual path to your file
try:
    df_raw = pd.read_csv(df_path, encoding='utf-16', sep='\t', decimal=',')

except FileNotFoundError:
    print("Error: 'your_raw_data.csv' not found. Please provide the correct path to your file.")
    # You might want to exit or handle this error differently
    exit()

# --- Data Preprocessing ---

# 1. Combine 'meas_ym', 'meas_dd', and 'id' into a single 'timestamp' column
# The 'id' column typically ranges from 1 to 96 for quarter-hour data.
# We'll need to adjust 'id' to represent minutes from start of the day.

#Drop supplies
NaN_Supplies_drop = ['SUPPLY018','SUPPLY019', 'SUPPLY082', 'SUPPLY094']
not_enough_days_drop = ['SUPPLY001', 'SUPPLY069', 'SUPPLY071', 'SUPPLY092']
df_raw = df_raw[~df_raw['Supply_ID'].isin(NaN_Supplies_drop + not_enough_days_drop)]

# Convert 'meas_ym' to string and 'meas_dd' to string for concatenation
df_raw['year_month_str'] = df_raw['meas_ym'].astype(str)
df_raw['day_str'] = df_raw['meas_dd'].astype(str).str.zfill(2) # Pad single digit days with a leading zero

# Create a 'date_part' string in 'YYYY-MM-DD' format
# 'meas_ym' is 'YYYYMM', so we need to insert a hyphen
df_raw['date_part'] = df_raw['year_month_str'].str[:4] + '-' + \
                      df_raw['year_month_str'].str[4:] + '-' + \
                      df_raw['day_str']

# Convert 'id' to minutes from start of day (id 1 = 00:00, id 96 = 23:45)
# (id - 1) * 15 gives minutes from midnight
df_raw['minutes_part'] = (df_raw['id'] - 1) * 15

# Create a temporary column for time strings (e.g., 'HH:MM')
# Use floor division for hours, modulo for remaining minutes
df_raw['time_part'] = (df_raw['minutes_part'] // 60).astype(str).str.zfill(2) + ':' + \
                      (df_raw['minutes_part'] % 60).astype(str).str.zfill(2) + ':00'

# Combine date and time parts into a full timestamp string
df_raw['timestamp_str'] = df_raw['date_part'] + ' ' + df_raw['time_part']

# Convert the combined string to datetime objects
df_raw['date'] = pd.to_datetime(df_raw['timestamp_str'])

# 2. Convert 'val' to numeric, handling potential non-numeric entries
df_raw['val'] = pd.to_numeric(df_raw['val'], errors='coerce') # 'coerce' will turn errors into NaN

# 3. Drop temporary columns if desired
df_raw = df_raw.drop(columns=['meas_ym', 'meas_dd', 'id', 'year_month_str',
                              'day_str', 'date_part', 'minutes_part', 'time_part', 'timestamp_str'])

# 4. Handle NaN values in 'val' (e.g., fill with 0 or the mean/median, or drop rows)
# For consumption data, filling with 0 for missing readings is often a reasonable approach.
df_raw = df_raw.dropna(subset=['val'])

# 5. Sort the data, essential for correct period slicing
df_raw = df_raw.sort_values(by=['Supply_ID', 'date']).reset_index(drop=True)

print("--- Processed Raw Data Head ---")
print(df_raw.head())
print(f"\nTotal rows in processed df_raw: {len(df_raw)}")
print(f"Number of unique Supply_IDs in processed df_raw: {df_raw['Supply_ID'].nunique()}")
print(f"Overall min timestamp in processed df_raw: {df_raw['date'].min()}")
print(f"Overall max timestamp in processed df_raw: {df_raw['date'].max()}")

# --- Diagnostic check for a sample supply after initial processing ---
sample_supply_id = df_raw['Supply_ID'].iloc[0] # Take the first supply ID
sample_supply_data = df_raw[df_raw['Supply_ID'] == sample_supply_id]
print(f"\n--- Diagnostic for {sample_supply_id} after processing ---")
print(f"Min date: {sample_supply_data['date'].min()}")
print(f"Max date: {sample_supply_data['date'].max()}")
print(f"Number of unique days: {sample_supply_data['date'].dt.date.nunique()}")
print(f"Total readings: {len(sample_supply_data)}")




def create_period_features_for_anomaly_detection(df, period_length_days=10, desired_total_days=400):
    """
    Transforms raw quarter-hour consumption data into a period-level feature dataset
    suitable for anomaly detection. It processes up to `desired_total_days` of data
    for each supply, starting from its earliest record.

    Args:
        df (pd.DataFrame): The input DataFrame with 'Supply_ID', 'val', and 'date' columns.
                           'date' should be in datetime format and contain timestamps.
        period_length_days (int): The length of each consumption period in days.
        desired_total_days (int): The maximum number of consecutive days of data to consider
                                  for each supply, starting from its first recorded date.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a period for a supply,
                      with extracted features.
    """

    all_period_features = []

    # Get unique supply IDs to process them individually, and ensure consistent order
    unique_supply_ids = df['Supply_ID'].unique()
    num_supplies = len(unique_supply_ids)
    processed_supplies_count = 0

    print(f"Starting feature generation for {num_supplies} supplies...")

    for supply_id in unique_supply_ids:
        supply_df = df[df['Supply_ID'] == supply_id].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        # Sort by date to ensure correct chronological order for slicing
        supply_df = supply_df.sort_values(by='date').reset_index(drop=True)

        # Determine the earliest actual date for this supply
        first_date_of_supply = supply_df['date'].min().normalize() # Normalize to just the date part

        # Identify the end date for the *desired_total_days* window
        # This defines the maximum range of data we want to process for this supply
        potential_end_date_for_supply_window = first_date_of_supply + pd.Timedelta(days=desired_total_days)

        # Filter the supply data to include only the first 'desired_total_days' worth of data
        # We use .normalize() to compare only dates, ignoring time components for day span calculation.
        # This means it will get all readings where the *date part* is within the desired window.
        supply_df_limited = supply_df[supply_df['date'].dt.normalize() < potential_end_date_for_supply_window].copy()


        # Check if there's enough data for at least one full period after limiting
        if supply_df_limited.empty:
            print(f"Warning: Supply {supply_id} has no data or less than 1 day of data available for processing. Skipping.")
            processed_supplies_count += 1
            continue

        # Recalculate actual first and last dates in the limited data
        actual_first_date_limited = supply_df_limited['date'].min().normalize()
        actual_last_date_limited = supply_df_limited['date'].max().normalize()
        actual_days_available_in_limited_data = (actual_last_date_limited - actual_first_date_limited).days + 1

        # Calculate the number of periods based on the actual available days (capped at desired_total_days)
        # We only create periods if there's enough data for a full period.
        num_periods_for_this_supply = actual_days_available_in_limited_data // period_length_days

        if num_periods_for_this_supply == 0:
            print(f"Warning: Supply {supply_id} has less than {period_length_days} days of data available ({actual_days_available_in_limited_data} days). Cannot form full periods. Skipping.")
            processed_supplies_count += 1
            continue

        # Calculate the total sum for ratio calculation *once* for the limited supply data
        supply_total_consumption_limited = supply_df_limited['val'].sum()

        for i in range(num_periods_for_this_supply):
            period_start = actual_first_date_limited + pd.Timedelta(days=i * period_length_days)
            period_end = period_start + pd.Timedelta(days=period_length_days)

            # Filter data for the current period, including full timestamps (>= start, < end)
            period_data = supply_df_limited[(supply_df_limited['date'] >= period_start) & (supply_df_limited['date'] < period_end)]

            if not period_data.empty:
                # --- Feature Engineering for the Current Period ---
                period_features = {
                    'Supply_ID': supply_id,
                    'Period_Start_Date': period_start,
                    'Period_End_Date': period_end,
                    'Period_Index': i + 1, # 1-based index for periods

                    # Basic statistics
                    'Mean_Consumption_Period': period_data['val'].mean(),
                    'Max_Consumption_Period': period_data['val'].max(),
                    'Min_Consumption_Period': period_data['val'].min(),
                    'Std_Consumption_Period': period_data['val'].std(),

                    # Robust statistics (less sensitive to extreme outliers)
                    'Median_Consumption_Period': period_data['val'].median(),
                    'IQR_Consumption_Period': period_data['val'].quantile(0.75) - period_data['val'].quantile(0.25),

                    # Features indicating unusual patterns
                    'Num_Zero_Readings_Period': (period_data['val'] == 0).sum(),
                    # These next two features require at least two days of data in the period to calculate a diff
                    'Daily_Avg_Change_Period': period_data.groupby(period_data['date'].dt.date)['val'].sum().diff().mean() if period_data['date'].dt.date.nunique() > 1 else np.nan,
                    'Daily_Std_Change_Period': period_data.groupby(period_data['date'].dt.date)['val'].sum().diff().std() if period_data['date'].dt.date.nunique() > 1 else np.nan,
                    'Max_Daily_Consumption': period_data.groupby(period_data['date'].dt.date)['val'].sum().max(),
                    'Min_Daily_Consumption': period_data.groupby(period_data['date'].dt.date)['val'].sum().min(),

                    # Trend of consumption within the period
                    'Trend_Period': np.polyfit(np.arange(len(period_data)), period_data['val'], 1)[0] if len(period_data) > 1 else 0,

                    # Ratio of period's total consumption to the supply's overall total consumption
                    'Period_Total_Consumption_Ratio_to_Supply_Total': period_data['val'].sum() / supply_total_consumption_limited if supply_total_consumption_limited > 0 else 0
                }
                all_period_features.append(period_features)
            else:
                # This warning means a period defined by date range has no data
                # This can happen if there are large data gaps *within* the 400-day window
                print(f"Warning: No data found for Supply {supply_id} Period {i+1} ({period_start} to {period_end}). This might indicate data gaps within the {desired_total_days}-day window.")
        
        processed_supplies_count += 1
        if processed_supplies_count % 10 == 0 or processed_supplies_count == num_supplies:
            print(f"Processed {processed_supplies_count}/{num_supplies} supplies...")


    return pd.DataFrame(all_period_features)

# --- Call the function with your processed df_raw ---
# Set period_length_days=10 and desired_total_days=400 as per your requirement
# This will process the first 400 days of data for each supply and divide them into 10-day periods.
period_features_df = create_period_features_for_anomaly_detection(
    df_raw,
    period_length_days=5,
    desired_total_days=1825
)

try:
    fraud_supply_labels_df = pd.read_csv(labels_path, encoding='utf-16', sep='\t', decimal=',')

except FileNotFoundError:
    print(f"Error: '{labels_path}' not found. Please provide the correct path to your labels file.")
    exit()

# Ensure Supply_ID in labels is consistent (uppercase, stripped spaces)
fraud_supply_labels_df['Supply_ID'] = fraud_supply_labels_df['Supply_ID'].astype(str).str.strip().str.upper()

# Create 'Is_Non_Regular' combined label
# 1 for 'Anomalous' or 'Fraud', 0 for 'Regular'
fraud_supply_labels_df['Is_Non_Regular'] = fraud_supply_labels_df['CLUSTER'].apply(
    lambda x: 1 if x in ['Anomalia', 'Frode'] else 0
)

print("Labeled Data Head (with Is_Non_Regular):")
print(fraud_supply_labels_df.head())
print(f"Counts of 'CLUSTER':\n{fraud_supply_labels_df['CLUSTER'].value_counts()}")
print(f"Counts of 'Is_Non_Regular':\n{fraud_supply_labels_df['Is_Non_Regular'].value_counts()}")


# Merge period features with the new 'Is_Non_Regular' label
merged_df = pd.merge(period_features_df, fraud_supply_labels_df[['Supply_ID', 'Is_Non_Regular']], on='Supply_ID', how='left')


print("\n--- Period-Level Feature Dataset Head ---")
print(merged_df.head())
print("\n--- Period-Level Feature Dataset Info ---")
merged_df.info()
print(f"\nTotal periods generated: {len(merged_df)}")
print(f"Number of unique supplies in the final dataset: {merged_df['Supply_ID'].nunique()}")

# Optional: Check number of periods per supply
if not merged_df.empty:
    print("\n--- Periods generated per supply (top 5 by count) ---")
    print(merged_df['Supply_ID'].value_counts().head())
    print("\n--- Periods generated per supply (bottom 5 by count) ---")
    print(merged_df['Supply_ID'].value_counts().tail())

#Save new dataset
merged_df.to_csv('/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Unsupervized/Data/full_dataset_5day.csv')
print('\nSaved dataset')