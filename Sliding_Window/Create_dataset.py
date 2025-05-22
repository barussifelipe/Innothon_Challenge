import pandas as pd


#Paths
path_consumption = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data/Consumption.csv'


def process_df(df, supplies_drop):

    # Combine 'meas_ym' and 'meas_dd' into a single date-like column for grouping
    df['date'] = pd.to_datetime(df['meas_ym'].astype(str) + df['meas_dd'].astype(str), format='%Y%m%d')

    # Sort by Supply_ID and date to ensure proper grouping
    df = df.sort_values(by=['Supply_ID', 'date'])

    #Drop old date columns
    df.drop(columns=['meas_ym', 'meas_dd', 'magnitude', 'id'], inplace=True)

    #Decrease to only last 400 days per supply
    df = df.groupby('Supply_ID').tail(400).reset_index(drop=True)

    #Get ride of missing vals supplies
    df = df[~df['Supply_ID'].isin(supplies_drop)]

    #Get ride of all 0s
    # Select only the feature columns (excluding 'Supply_ID' and any non-feature columns)
    feature_columns = [col for col in df.columns if col.startswith('mean_') or col.startswith('max_') or col.startswith('min_') or col.startswith('sd_')]

    # Find rows where all feature values are zero
    rows_with_all_zeros = df[df[feature_columns].eq(0).all(axis=1)]

    # Print the rows with all zeros
    all_zeroes = list(rows_with_all_zeros['Supply_ID'])

    #Drop these rows
    # Drop rows where Supply_ID is in the all_zeroes list
    df = df[~df['Supply_ID'].isin(all_zeroes)]

    return df
    
def create_features_by_x_days(df, x):
    """
    Create a new dataset with aggregated features grouped by Supply_ID and every x days.

    Args:
        df (pd.DataFrame): The input dataset with columns ['Supply_ID', 'date', 'val'].
        x (int): The number of days to group by.

    Returns:
        pd.DataFrame: A new dataset with aggregated features for every x days.
    """
    # Create a new column to group by every x days
    df['x_day_group'] = df.groupby('Supply_ID').cumcount() // x

    # Initialize an empty dictionary to store features for each Supply_ID
    feature_data = {}

    # Group by Supply_ID and x_day_group
    grouped = df.groupby(['Supply_ID', 'x_day_group'])

    for (supply_id, group_id), group in grouped:
        # Calculate the required statistics for the 'val' column
        mean_val = group['val'].mean()
        max_val = group['val'].max()
        min_val = group['val'].min()
        std_val = group['val'].std()

        # Add the features to the dictionary for the current Supply_ID
        if supply_id not in feature_data:
            feature_data[supply_id] = []
        feature_data[supply_id].extend([mean_val, max_val, min_val, std_val])

    # Create a DataFrame from the feature dictionary
    feature_columns = []
    for i in range(1, (df['x_day_group'].max() + 2)):  # +2 to account for 0-based indexing
        feature_columns.extend([f'mean_x{i}_days', f'max_x{i}_days', f'min_x{i}_days', f'sd_x{i}_days'])

    result_df = pd.DataFrame.from_dict(feature_data, orient='index', columns=feature_columns).reset_index()
    result_df.rename(columns={'index': 'Supply_ID'}, inplace=True)

    return result_df

def study_df(df):

    #NaN values

    #NaN values per column
    nan_per_column = df.isna().sum()
    print(f'\nNaN values per column:\n{nan_per_column}')

    #Get ride of all 0s
    # Select only the feature columns (excluding 'Supply_ID' and any non-feature columns)
    feature_columns = [col for col in df.columns if col.startswith('mean_') or col.startswith('max_') or col.startswith('min_') or col.startswith('sd_')]

    # Find rows where all feature values are zero
    rows_with_all_zeros = df[df[feature_columns].eq(0).all(axis=1)]

    # Print the rows with all zeros
    all_zeroes = list(rows_with_all_zeros['Supply_ID'])

    return

# Example usage
if __name__ == '__main__':


    # supplies_drop = ['SUPPLY019', 'SUPPLY082', 'SUPPLY094']
    # #Read in consumption data
    # df_consumption = pd.read_csv(path_consumption, encoding='utf-16', sep='\t', decimal=',')
    # #Save base df
    # process_df(df_consumption, supplies_drop).to_csv('/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Sliding_Window/base_dataset400.csv')
    # print('saved')

    #Base df 
    base_df_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Sliding_Window/base_dataset400.csv'
    base_df = pd.read_csv(base_df_path)

    # Create features grouped by every 3 days
    x_days = 10
    new_df = create_features_by_x_days(base_df, x_days)

    #Save new dataset
    new_df.to_csv('/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Sliding_Window/sliding_window_dataset.csv')

    print(new_df.head())



