import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler



input_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data/Consumption.csv'

#Read into dataframe
df_consumption = pd.read_csv(input_path, encoding='utf-16', sep='\t', decimal=',')

# #Overral info
# print(f'\nOverral info:\n')
# df_consumption.info()

# #Magnitude balance
# mag_bal = df_consumption['magnitude'].value_counts()
# print(f'\nMagnitude balance:\n{mag_bal}')

# #NaN values per column
# nan_per_column = df_consumption.isna().sum()
# print(f'\nNaN values per column:\n{nan_per_column}')

# #NaN val (kwh) per supply
# rows_with_nan = df_consumption[df_consumption.isna().any(axis=1)]
# val_nan_by_supply = rows_with_nan.groupby(['Supply_ID']).size()
# print(f'\nVal (kwh) Nan values per supply:\n{val_nan_by_supply}')

#Number of days logged per supply (Assumes every day is complete (if not each supply might be off by one day))
filter_df = df_consumption[['id', 'Supply_ID']]
days_logged_per_supply = filter_df[filter_df['id'] == 96].groupby(['Supply_ID']).size()

min_days_logged = min(days_logged_per_supply)
max_days_logged = max(days_logged_per_supply)
mean_days_logged = days_logged_per_supply.mean()
mode_days_logged = days_logged_per_supply.mode()

#Only supplies that contain x amount of days
x = 1825
valid_supplies = days_logged_per_supply[days_logged_per_supply < x].index.tolist()


print(f'number of supplies with at least {x} days: {len(valid_supplies)}')
print(valid_supplies)

# # Create a histogram for days logged per supply
# plt.figure(figsize=(8, 6))
# plt.hist(days_logged_per_supply, bins=20, color='lightblue', edgecolor='black')

# # Add labels and title
# plt.title('Histogram of Days Logged per Supply', fontsize=14)
# plt.xlabel('Number of Days Logged', fontsize=12)
# plt.ylabel('Frequency', fontsize=12)

# #Save image
# save_img_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Study_datasets/Consumption/plots/Distribution_days_logged_hist.png'
# plt.savefig(save_img_path)
# print(f'Distribution of Days Logged per Supply plot saved to: {save_img_path}')

# # Show the plot
# plt.show()

print(f'\nNumber of days logged per supply\nMinimum days logged: {min_days_logged}\nMaximum days logged: {max_days_logged}\nAverage days logged: {mean_days_logged}\nMode of days logged: {mode_days_logged.values}')

