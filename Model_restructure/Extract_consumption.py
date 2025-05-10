import pandas as pd

#Consumption dataset path
main_df_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data/Consumption.csv'
grouped_df_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Model_restructure/Data/consumption_data.csv'


def group_compute_stats(input_path):
    # Load in consumption dataset
    df = pd.read_csv(input_path, encoding='utf-16', sep='\t', decimal=',')

    # Filter df to contain only Supply_ID, val, and meas_ym
    df = df[['Supply_ID', 'meas_ym', 'val']]

    # Get rid of month in year-month column (only include first 4 characters)
    df['meas_ym'] = df['meas_ym'].astype(str).str[:4]

    # Group data by Supply_ID and meas_ym, and calculate multiple statistics for 'val'
    grouped = df.groupby(['Supply_ID', 'meas_ym'])['val'].agg(['mean', 'max', 'min', 'std']).reset_index()

    # Rename columns for clarity
    grouped.rename(columns={
        'mean': 'val_mean',
        'max': 'val_max',
        'min': 'val_min',
        'std': 'val_std'
    }, inplace=True)

    return grouped


def treat_grouped(input_path):

    df = pd.read_csv(input_path)

    #Dataframe info 
    print(df.info())

    #Check how many entries per supply
    entries_per_supply = df.groupby(['Supply_ID']).size()
    print(f'\nentries per supply: {entries_per_supply}\n')
    
    #Max, min and average number of entries
    print(f'max entries: {max(entries_per_supply)}\n')
    print(f'min entries: {min(entries_per_supply)}\n')
    print(f'mean entries: {entries_per_supply.mean()}\n')


    return










if __name__ == '__main__':
    
    # #Group and compute stats
    # grouped = group_compute_stats(main_df_path)
    
    # #Save grouped data for easier use later
    # grouped.to_csv('/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Model_restructure/Data/consumption_data.csv', index=False)

    treat_grouped(grouped_df_path)



    None