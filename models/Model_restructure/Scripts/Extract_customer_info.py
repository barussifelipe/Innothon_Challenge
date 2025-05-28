import pandas as pd
from datetime import datetime, timedelta

# Main df file path
main_df_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data/Customer_info.csv'
output_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Model_restructure/Processed_customer_info.csv'

# Read main df
print("Reading the input file...")
df = pd.read_csv(main_df_path, encoding='utf-16', sep='\t', decimal=',')
print(f"Input file loaded. Number of rows: {len(df)}")

# Convert date columns to datetime
print("Converting date columns to datetime...")
df['begin_date_ref'] = pd.to_datetime(df['begin_date_ref'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
df['end_date_ref'] = pd.to_datetime(df['end_date_ref'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
print("Date columns converted.")

# Handle NaN values in available_power
print("Handling NaN values in available_power...")
df['available_power'] = df['available_power'].fillna(0)
print("NaN values handled.")

#Check Invalid rows
invalid_rows = df[df['end_date_ref'] < df['begin_date_ref']]
print("Invalid rows where end_date_ref is earlier than begin_date_ref:")
print(invalid_rows)

# Filter out rows where end_date_ref is earlier than begin_date_ref
print("Filtering out invalid rows where end_date_ref is earlier than begin_date_ref...")
df = df[df['end_date_ref'] >= df['begin_date_ref']]
print(f"Number of valid rows after filtering: {len(df)}")

# Vectorized function to split date ranges into yearly intervals
def split_into_years_vectorized(df):
    print("Splitting date ranges into yearly intervals (vectorized)...")
    df['begin_year'] = df['begin_date_ref'].dt.year
    df['end_year'] = df['end_date_ref'].dt.year
    years = (
        df.loc[df.index.repeat(df['end_year'] - df['begin_year'] + 1)]
        .assign(Year=lambda x: x.groupby(level=0).cumcount() + x['begin_year'])
    )
    return years.drop(columns=['begin_year', 'end_year'])

# Apply the vectorized function
expanded_df = split_into_years_vectorized(df)
print(f"Expanded DataFrame created. Number of rows: {len(expanded_df)}")

# Group by Supply_ID and Year, calculate mean available_power and determine status
def determine_status(statuses):
    unique_statuses = set(statuses)
    if len(unique_statuses) > 1:
        return 'M'  # Mixed
    elif 'A' in unique_statuses:
        return 'A'  # Active
    elif 'C' in unique_statuses:
        return 'C'  # Terminated
    elif 'F' in unique_statuses:
        return 'F'  # Fictitious
    else:
        return 'Unknown'

print("Grouping data by Supply_ID and Year...")
final_df = (
    expanded_df.groupby(['Supply_ID', 'Year'])
    .agg(
        mean_available_power=('available_power', 'mean'),
        Status=('supply_status', determine_status)
    )
    .reset_index()
)

# Round the mean_available_power column to 2 decimal places
final_df['mean_available_power'] = final_df['mean_available_power'].round(3)
print("Data grouped and aggregated.")

# Encode the Status column
print("Encoding the Status column...")
status_mapping = {'A': 0, 'C': 1, 'F': 2, 'M': 3, 'Unknown': -1}
final_df['Status_encoded'] = final_df['Status'].map(status_mapping)
print("Status column encoded.")

# # Save the final DataFrame to a CSV file
# print(f"Saving the final DataFrame with encoded Status to {output_path}...")
# final_df.to_csv(output_path, index=False)
# print("Final DataFrame saved.")

# Print the first few rows of the final DataFrame
print("Here are the first few rows of the final DataFrame with encoded Status:")
print(final_df.head(15))


#Info on dataset


#Dataframe info 
print(final_df.info())

#Check how many entries per supply
entries_per_supply = final_df.groupby(['Supply_ID']).size()
print(f'\nentries per supply: {entries_per_supply}\n')

#Max, min and average number of entries
print(f'max entries: {max(entries_per_supply)}\n')
print(f'min entries: {min(entries_per_supply)}\n')
print(f'mean entries: {entries_per_supply.mean()}\n')

