import pandas as pd

#Consumption dataset path
input_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data/Consumption.csv'

#Load in consumption dataset
df = pd.read_csv(input_path, encoding='utf-16', sep='\t', decimal=',')

print(df['meas_ym'].head())