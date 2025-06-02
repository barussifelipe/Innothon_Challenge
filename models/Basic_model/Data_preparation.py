import pandas as pd

#Define paths
consumption_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data/Consumption.csv'
labels_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data/Labels.csv'

#Function to take only a sample of the dataset for training
def sample_dataset_by_chunks(input_path, value_to_count, max_appearances, groupby_column, filter_column, chunksize=10000):
    """
    Processes the dataset in chunks to include all rows up to the max_appearances of a specific value in a column, grouped by another column.

    Args:
        input_path (str): Path to the input CSV file.
        value_to_count (int): The value to count in the filter_column (e.g., 96).
        max_appearances (int): The maximum number of appearances to include (e.g., 300).
        groupby_column (str): The column to group by (e.g., 'Supply_ID').
        filter_column (str): The column to count the value in (e.g., 'id').
        chunksize (int): Number of rows to process per chunk.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    # Dictionary to track cumulative counts for each group
    cumulative_counts = {}

    # Initialize an empty list to store filtered chunks
    filtered_chunks = []

    # Read the file in chunks
    for chunk in pd.read_csv(input_path, encoding='utf-16', sep='\t', decimal=',', chunksize=chunksize):
        # Iterate over each group in the chunk
        for group_name, group_data in chunk.groupby(groupby_column):
            # Initialize the cumulative count for the group if not already done
            if group_name not in cumulative_counts:
                cumulative_counts[group_name] = 0

            # Create a cumulative count of the value_to_count in the filter_column
            group_data['cumulative_count'] = (group_data[filter_column] == value_to_count).cumsum() + cumulative_counts[group_name]

            # Update the cumulative count for the group
            cumulative_counts[group_name] = group_data['cumulative_count'].iloc[-1]

            # Filter rows where the cumulative count is less than or equal to max_appearances
            group_filtered = group_data[group_data['cumulative_count'] <= max_appearances]

            # Drop the temporary 'cumulative_count' column
            group_filtered = group_filtered.drop(columns=['cumulative_count'])

            # Append the filtered group to the list
            filtered_chunks.append(group_filtered)

    # Concatenate all filtered chunks into a single DataFrame
    filtered_df = pd.concat(filtered_chunks, ignore_index=True)

    return filtered_df

#Function to get ride of missing vals supplies
def drop_NaN_supplies(df, supplies_drop):

    ret = df[~df['Supply_ID'].isin(supplies_drop)]
    print(f'Dropped supplies: {supplies_drop}')
    return ret

#Function to segment dataset
def segment_dataset(df, number_segments):
    
    #Segments should contain in how many segments the 96 quarter hour intervals to be divided (24 segments for hourly)

    #Dictionary to store segments
    segments = {}
    
    #Individual segment size
    segment_size = 96//number_segments

    for i in range(1, number_segments + 1):
        
        #Define beginning and end of each segment
        seg_begin = segment_size * (i - 1)
        seg_end = segment_size * i
        
        #Update dictionary with segments
        segments[f'Segment {i}'] = df[df['id'].isin(range(seg_begin, seg_end))]

    return segments

#Function to compute segment averages
def segmented_averages(segments):
    #Now for each segment I need to compute the average of each group

    averages = {}

    for segment_num, segment_data in segments.items():

        #Compute averages of each supply for current segment
        current_seg_avg = segment_data.groupby(['Supply_ID'])['val'].mean().round(4)
        
        #Upadate dictionary
        averages[segment_num] = current_seg_avg

    return averages

#Function to convert to data frame
def prepare_dataframe(averages):
    """
    Combines segment averages into a single DataFrame.

    Args:
        averages (dict): A dictionary where keys are segment names (e.g., 'Segment 1') 
                         and values are DataFrames with averages grouped by Supply_ID.

    Returns:
        pd.DataFrame: A DataFrame where the first column is Supply_ID and the rest are 
                      segment averages.
    """
    # Initialize an empty DataFrame
    combined_df = pd.DataFrame()

    # Iterate over each segment and its averages
    for segment_num, segment_data in averages.items():
        # Reset index to bring Supply_ID as a column
        segment_data = segment_data.reset_index()

        # Rename the 'val' column to the segment name
        segment_data.rename(columns={'val': segment_num}, inplace=True)

        # Merge with the combined DataFrame
        if combined_df.empty:
            combined_df = segment_data
        else:
            combined_df = pd.merge(combined_df, segment_data, on='Supply_ID', how='outer')

    return combined_df

#Function to combine feature dataframe with labels
def combine_features_labels(df, supplies_drop):
    #Read labels dataset
    labels = pd.read_csv(labels_path, encoding='utf-16', sep='\t', decimal=',')

    #Drop unused Supplies
    labels = labels[~labels['Supply_ID'].isin(supplies_drop)]

    #Encode labels (1 if regular, 0 if not)
    labels['CLUSTER'] = labels['CLUSTER'].replace({'Frode': 0, 'Anomalia': 0, 'Regolare':1}).astype(int)

    #Merge features and labels
    merged_df = pd.merge(df, labels, on=['Supply_ID'], how='outer')

    #Rename Label column
    merged_df = merged_df.rename(columns={'CLUSTER':'is_Regular'})
    
    #Set Supply_ID as index
    merged_df.set_index('Supply_ID', inplace=True)
    
    return merged_df


if __name__ == '__main__':

    # Define the list of supplies you want to include
    # Example: selected_supplies = ['SUPPLY001', 'SUPPLY005', 'SUPPLY010']
    # If you want to include all supplies, set selected_supplies = None or an empty list
    selected_supplies = ['SUPPLY_006', 'SUPPLY_002', 'SUPPLY_030', 'SUPPLY_1O0'] # Set this to a list of Supply_IDs if you want to filter

    #Sample the dataset
    filtered_df = sample_dataset_by_chunks(
        input_path=consumption_path,
        value_to_count=96,
        max_appearances=1825,
        groupby_column='Supply_ID',
        filter_column='id',
        chunksize=10000
    )

    # Filter by selected supplies if the list is provided
    if selected_supplies is not None and len(selected_supplies) > 0:
        df_consumption = filtered_df[filtered_df['Supply_ID'].isin(selected_supplies)].copy()
        print(f"Filtered to include only selected supplies: {selected_supplies}")
    else:
        df_consumption = filtered_df.copy()


    #Get ride of missing vals supplies (these are still dropped even if you select supplies)
    supplies_to_drop = ['SUPPLY019', 'SUPPLY082', 'SUPPLY094']
    df_consumption = drop_NaN_supplies(df_consumption, supplies_to_drop)


    #Segment the dataset
    segments = segment_dataset(
        df_consumption,
        number_segments= 96 #Define number of segments to divide day
    )

    #Compute averages
    segments_averages = segmented_averages(segments)

    #Convert to dataframe
    dataset_df = prepare_dataframe(segments_averages)
    
    #Merge features and labels
    final_df = combine_features_labels(dataset_df, supplies_to_drop)
    print(final_df.head())

    #Save final df as csv
    output_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Study_datasets/Consumption'
    final_df.to_csv(f'{output_path}/Basic_model_dataset_quarterhouravg_1825days_selected_supplies.csv')

'''
To select days: I know each day is 'marked by a 96' i wanna take the rows up to the 300th appearance
of a 96 for each supply
'''