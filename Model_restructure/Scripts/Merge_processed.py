import pandas as pd


#Extracted dataset paths
consumption_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Model_restructure/Data/consumption_data.csv'
customer_info_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Model_restructure/Data/Processed_customer_info.csv'
interruptions_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Study_datasets/Interruptions/Interruptions_processed.csv'
works_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Study_datasets/Work/Work_processed.csv'

def dataset_info(df):

    #Dataframe info 
    print(df.info())

    #Check how many entries per supply
    entries_per_supply = df.groupby(['Supply_ID']).size()
    print(f'\nentries per supply: {entries_per_supply}\n')

    # #Check range of years per supply
    # unique_years_supply = df.groupby(['Supply_ID'])['meas_ym'].unique()
    # print(f'Unique years by supply:\n{unique_years_supply}\n')

    #Max, min and average number of entries
    print(f'max entries: {max(entries_per_supply)}\n')
    print(f'min entries: {min(entries_per_supply)}\n')
    print(f'mean entries: {entries_per_supply.mean()}\n')

    #NaN values
    #NaN values per column
    nan_per_column = df.isna().sum()
    print(f'\nNaN values per column:\n{nan_per_column}')
    #Non nan values per column
    non_nan_per_column = (~df.isna()).sum()
    print(f'\nNon NaN values per column:\n{non_nan_per_column}')
    

    return

#Each dataset must contain 6 entries per supply. Fill 0s inn case needed to complete 6 entries(years)

#Consumption
def consumption_ensure_six_years(df):
    # Define the target number of years per supply
    target_years = 6
    max_year = 2024  # Ensure no years exceed 2024

    # Create an empty list to store the updated rows
    updated_rows = []

    # Group by Supply_ID
    grouped = df.groupby('Supply_ID')

    for supply_id, group in grouped:
        # Get the existing years for the current supply
        existing_years = set(group['meas_ym'])

        # Determine the range of years to fill
        min_year = min(existing_years)
        all_years = set(range(min_year, max_year + 1)).union(existing_years)

        # Add missing years to reach the target number of years
        while len(all_years) < target_years:
            if max(all_years) < max_year:
                all_years.add(max(all_years) + 1)  # Add future years if possible
            else:
                all_years.add(min(all_years) - 1)  # Add past years if future years exceed 2024

        # Create rows for missing years
        for year in sorted(all_years):
            if year not in existing_years:
                updated_rows.append({
                    'Supply_ID': supply_id,
                    'meas_ym': year,
                    'val_mean': None,
                    'val_max': None,
                    'val_min': None,
                    'val_std': None
                })

        # Add the existing rows
        updated_rows.extend(group.to_dict('records'))

    # Create a new DataFrame with the updated rows
    updated_df = pd.DataFrame(updated_rows)

    # Sort by Supply_ID and meas_ym
    updated_df = updated_df.sort_values(by=['Supply_ID', 'meas_ym']).reset_index(drop=True)

    #Rename year column
    updated_df.rename(columns={'meas_ym': 'Year'}, inplace=True)

    print(f'columns for updated_consumption:\n{updated_df.columns}')

    return updated_df

def extract_years(df):
    #Check range of years per supply
    unique_years_supply = df.groupby(['Supply_ID'])['Year'].unique().apply(list).to_dict()
    #print(f'Unique years by supply:\n{unique_years_supply}\n')

    return unique_years_supply

def customer_info_extract_and_fill_years(df, years_dict):
    """
    Extract rows for specified years and add empty rows for missing years.

    Args:
        df (pd.DataFrame): The input dataset.
        years_dict (dict): A dictionary where keys are Supply_IDs and values are lists of years.

    Returns:
        pd.DataFrame: A new dataset with rows for the specified years.
    """
    # Create an empty list to store the updated rows
    updated_rows = []

    # Iterate over each Supply_ID and its corresponding years
    for supply_id, years in years_dict.items():
        # Filter rows for the current Supply_ID
        supply_rows = df[df['Supply_ID'] == supply_id]

        # Check for missing years
        existing_years = set(supply_rows['Year'])
        missing_years = set(years) - existing_years

        # Add rows for missing years
        for year in missing_years:
            updated_rows.append({
                'Supply_ID': supply_id,
                'Year': year,
                'mean_available_power': None,
                'Status': None,
                'Status_encoded': None
            })

        # Add the existing rows for the specified years
        filtered_rows = supply_rows[supply_rows['Year'].isin(years)].to_dict('records')
        updated_rows.extend(filtered_rows)

    # Create a new DataFrame with the updated rows
    updated_df = pd.DataFrame(updated_rows)

    # Remove duplicates (if any)
    updated_df = updated_df.drop_duplicates(subset=['Supply_ID', 'Year']).reset_index(drop=True)

    # Enforce exactly 6 entries per Supply_ID
    final_rows = []
    grouped = updated_df.groupby('Supply_ID')
    for supply_id, group in grouped:
        if len(group) > 6:
            # If more than 6 entries, keep only the first 6
            final_rows.extend(group.head(6).to_dict('records'))
        elif len(group) < 6:
            # If less than 6 entries, add empty rows for missing years
            existing_years = set(group['Year'])
            all_years = set(years_dict[supply_id])
            missing_years = list(all_years - existing_years)
            for year in sorted(missing_years)[:6 - len(group)]:
                final_rows.append({
                    'Supply_ID': supply_id,
                    'Year': year,
                    'mean_available_power': None,
                    'Status': None,
                    'Status_encoded': None
                })
            final_rows.extend(group.to_dict('records'))
        else:
            # If exactly 6 entries, keep as is
            final_rows.extend(group.to_dict('records'))

    # Create the final DataFrame
    final_df = pd.DataFrame(final_rows)

    # Sort by Supply_ID and Year
    final_df = final_df.sort_values(by=['Supply_ID', 'Year']).reset_index(drop=True)

    return final_df


def merge_consumption_customer(consumption, customer_info):
    # Merge the datasets on Supply_ID and Year
    merged_df = pd.merge(consumption, customer_info, on=['Supply_ID', 'Year'], how='inner')

    return merged_df


if __name__ == '__main__':

    consumption_df = pd.read_csv(consumption_path)
    customer_info_df = pd.read_csv(customer_info_path)

    #Add empty rows
    updated_consumption_df = consumption_ensure_six_years(consumption_df)

    #Extract years to be used in all datasets
    years_extracted = extract_years(updated_consumption_df)

    #Update customer_info accordingly
    updated_customer_info_df = customer_info_extract_and_fill_years(customer_info_df, years_extracted)

    #Merge dataframes
    merged_df = merge_consumption_customer(updated_consumption_df, updated_customer_info_df)

    # Print the merged dataset
    print(merged_df.head(10))

    #Info on merged dataset
    dataset_info(merged_df)


    # # Save the merged dataset to a CSV file
    # merged_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Model_restructure/Data/Merged_dataset.csv'
    # merged_df.to_csv(merged_path, index=False)


