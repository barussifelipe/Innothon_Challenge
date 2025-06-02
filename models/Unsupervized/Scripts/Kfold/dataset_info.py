import pandas as pd

input_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Unsupervized/Data/full_dataset_5day.csv'

df = pd.read_csv(input_path)

print(len(df['Supply_ID'].unique()))