import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

'''

Supply_ID	meas_ym	meas_dd	id	val	magnitude
SUPPLY001	202210	20	1	0	A1
SUPPLY001	202210	20	2	0	A1
SUPPLY001	202210	20	3	0	A1
SUPPLY001	202210	20	4	0	A1
SUPPLY001	202210	20	5	0	A1
SUPPLY001	202210	20	6	0	A1
SUPPLY001	202210	20	7	0	A1
SUPPLY001	202210	20	8	0	A1
SUPPLY001	202210	20	9	0	A1
SUPPLY001	202210	20	10	0	A1
SUPPLY001	202210	20	11	0	A1

'''

#Path
CONSUMPTION_PATH = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data/Consumption.csv'
PROCESSED_CONSUMPTION_PATH = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Basic_model/Data/Basic_model_dataset_quarterhouravg_400days.csv'
PLOTS_PATH = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Study_datasets/Consumption/plots'

#Load in dataset
df_consumption = pd.read_csv(CONSUMPTION_PATH, encoding='utf-16', sep='\t', decimal=',')

def count_supplies_with_x_complete_days(df: pd.DataFrame, x: int) -> int:
    """
    Counts the number of supplies that have exactly 'x' complete days of data.

    A 'complete day' is defined as having 96 quarter-hour records for a
    specific Supply_ID on a given date.

    Args:
        df (pd.DataFrame): The input DataFrame with columns 'Supply_ID',
                           'meas_ym', 'meas_dd', and 'id'.
        x (int): The target number of complete days to count supplies for.

    Returns:
        int: The number of supplies that have exactly 'x' complete days.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input 'df' must be a Pandas DataFrame.")
        return 0
    if not all(col in df.columns for col in ['Supply_ID', 'meas_ym', 'meas_dd', 'id']):
        print("Error: DataFrame must contain 'Supply_ID', 'meas_ym', 'meas_dd', and 'id' columns.")
        return 0
    if not isinstance(x, int) or x < 0:
        print("Error: 'x' must be a non-negative integer.")
        return 0

    # Combine year-month and day to create a unique date identifier for grouping
    # Using .astype(str) ensures consistent string concatenation, especially if meas_dd is int
    df['full_date'] = df['meas_ym'].astype(str) + '-' + df['meas_dd'].astype(str)

    # Group by Supply_ID and the combined date, then count the number of records (ids)
    # for each supply on each day.
    daily_record_counts = df.groupby(['Supply_ID', 'full_date'])['id'].count()

    # Identify complete days: a day is complete if it has exactly 96 records.
    # We reset the index to turn the grouped Series back into a DataFrame,
    # which makes it easier to work with 'Supply_ID' as a column.
    complete_days_df = daily_record_counts[daily_record_counts == 96].reset_index()

    # Now, group by Supply_ID again to count how many complete days each supply has.
    # .size() is used here to get the count of rows in each group.
    supply_complete_day_counts = complete_days_df.groupby('Supply_ID').size()

    # Finally, count how many supplies have exactly 'x' complete days.
    # We create a boolean Series where True indicates a supply has 'x' complete days,
    # and then sum the True values (which are treated as 1).
    num_supplies = (supply_complete_day_counts >= x).sum()

    return num_supplies



def plot_avgs(df, supplies_to_plot):
    """
    Plots the segment values for each specified supply on a single graph.
    Lines are colored blue if 'is_Regular' is 1 and red if 'is_Regular' is 0.

    Args:
        df (pd.DataFrame): The DataFrame containing Supply_ID, segment values, and 'is_Regular' feature.
        supplies_to_plot (list): A list of Supply_IDs to plot.
    """
    sns.set_style("whitegrid") # Set a style for better aesthetics
    plt.figure(figsize=(14, 7)) # Create a single figure for all plots

    for supply_id in supplies_to_plot:
        # Filter the DataFrame for the current supply
        current_supply_data = df[df['Supply_ID'] == supply_id]

        if current_supply_data.empty:
            print(f"No data found for Supply_ID: {supply_id}")
            continue

        # Get the 'is_Regular' value for the current supply
        # Use .iloc[0] to get the value from the first (and likely only) row
        is_regular_value = current_supply_data['is_Regular'].iloc[0]

        # Determine the color based on 'is_Regular'
        line_color = 'blue' if is_regular_value == 1 else 'red'
        line_label = f"{supply_id} (Regular)" if is_regular_value == 1 else f"{supply_id} (Anomaly/Fraud)"


        # Extract segment columns. Assuming segments are named 'Segment 1', 'Segment 2', etc.
        # We'll drop 'Supply_ID' and 'is_Regular' to get just the segment values
        segment_values = current_supply_data.drop(columns=['Supply_ID', 'is_Regular'], errors='ignore')

        # Transpose to get segments on the x-axis and values on the y-axis
        segment_values_transposed = segment_values.iloc[0].T

        # Plot on the same axes with the determined color and updated label
        if is_regular_value == 1:
            a = 0.1
        else:
            a = 1

        plt.plot(segment_values_transposed.index, segment_values_transposed.values,
                 marker='o', linestyle='-', color=line_color, alpha=a)

    plt.title('Average Consumption Across Quarters for Selected Supplies (Regular: Blue, Anomaly/Fraud: Red)')
    plt.xlabel('Quarter-Hour Interval')
    plt.ylabel('Average Consumption Value')
    plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for readability
    plt.legend(title='Supply ID & Status', loc='best') # Add a legend to distinguish between supplies and their status
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig(os.path.join(PLOTS_PATH, 'Average_quarter_consumption.png'))
    plt.show()


def plot_group_averages(df):
    """
    Plots the average segmented consumption for regular and non-regular supplies.

    Args:
        df (pd.DataFrame): The DataFrame containing Supply_ID, segment values, and 'is_Regular' feature.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 7))

    # Identify segment columns
    segment_cols = [col for col in df.columns if 'Segment' in col]

    # Calculate average for Regular supplies (is_Regular == 1)
    regular_avg = df[df['is_Regular'] == 1][segment_cols].mean()
    plt.plot(regular_avg.index, regular_avg.values, color='blue', linestyle='-',
             linewidth=2, label='Average Regular Supply')

    # Calculate average for Non-Regular supplies (is_Regular == 0)
    non_regular_avg = df[df['is_Regular'] == 0][segment_cols].mean()
    plt.plot(non_regular_avg.index, non_regular_avg.values, color='red', linestyle='-',
             linewidth=2, label='Average Anomaly/Fraud Supply')

    plt.title('Average Quarter Consumption: Regular vs. Anomaly/Fraud')
    plt.xlabel('Quarter-Hour Interval')
    plt.ylabel('Average Consumption Value')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Supply Type')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'Average_quarter_consumption_Group_average.png'))
    plt.show()

df_processed_consumption = pd.read_csv(PROCESSED_CONSUMPTION_PATH)
supplies = df_processed_consumption['Supply_ID'].unique()

plot_group_averages(df_processed_consumption)
plot_avgs(df_processed_consumption, supplies)

#Number of entries per supply

# supply_23 = df_consumption[df_consumption['Supply_ID'] == 'SUPPLY023']

# print(f'Supply_023 has {len(supply_23)} entries')


# x = 1825
# print(f'Supplies with at least {x} days: {count_supplies_with_x_complete_days(df_consumption, x)}')