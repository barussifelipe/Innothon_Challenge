import pandas as pd

#Dataset path
customer_info_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data/Customer_info.csv'

#Convert to dataframe
df_customer_info = pd.read_csv(customer_info_path, encoding='utf-16', sep='\t', decimal=',')


# Check supply status balance
ss_balance = df_customer_info['supply_status'].value_counts()
print(ss_balance)

#Look for missing data (missing available power in active cases)
df_ci_active = df_customer_info[df_customer_info['supply_status'] == 'A']
nan_count = df_ci_active['available_power'].isna().sum()
print(f'Number of available_power NaN in active cases {nan_count}')


#Logs per customer

log_per_customer = df_customer_info.groupby(['Supply_ID']).size()
max_lpc = max(log_per_customer)
min_lpc = min(log_per_customer)
mean_lpc = log_per_customer.mean()

print(f'\nLogs per supply info:\n')
print(f'Maximum number of logs of a supply: {max_lpc}')
print(f'Minimum number of logs of a supply: {min_lpc}')
print(f'Average number of logs of a supply: {mean_lpc}')


#Year ranges

#Understand range of years for each supply

#First log date and time of each supply
first_logs = df_customer_info.groupby(['Supply_ID'])['begin_date_ref'].first()
last_log = df_customer_info.groupby(['Supply_ID'])['end_date_ref'].last()
second_last_log = df_customer_info.groupby(['Supply_ID'])['end_date_ref'].nth(-2)

#Year specific
first_year = first_logs.str[6:10]
last_year = last_log.str[6:10]
second_last_year = second_last_log.str[6:10]

oldest_first_log_year = min(first_year)
newest_first_log_year = max(first_year)
oldest_last_log_year = min(last_year)
newest_last_log_year = max(last_year)

print(f'\nOldest first log: {oldest_first_log_year}\nNewest first log: {newest_first_log_year}\nOldest last log year: {oldest_last_log_year}\nNewest last log year: {newest_last_log_year}')


