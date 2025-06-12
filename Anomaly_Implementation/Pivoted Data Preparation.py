import numpy as np
import pandas as pd 

#Raw quarter hour consumption data

# Supply_ID	meas_ym	meas_dd	id	val	magnitude
# SUPPLY001	202210	20	1	0	A1
# SUPPLY001	202210	20	2	0	A1
# SUPPLY001	202210	20	3	0	A1
# SUPPLY001	202210	20	4	0	A1
# SUPPLY001	202210	20	5	0	A1
# SUPPLY001	202210	20	6	0	A1
# SUPPLY001	202210	20	7	0	A1


consumption_data = pd.read_csv("Study_datasets/EsempioDataset - CONSUMI.csv", encoding="utf-16", sep="\t", decimal = ',', header=0)

#Now I want to change the format of the data, I want to have a column for each supply ID and each row as a time point.

# First of all, I want to combine my meas_ym, which is 202210, 2022(year)/10(month) in this format and then give concatenate in a new column with meas_dd so I will have a date column in the format YYYY/MM/DD.

consumption_data["meas_ym"] = consumption_data['meas_ym'].astype(str).str[:4] + '/' + consumption_data['meas_ym'].astype(str).str[4:6]
consumption_data['meas_dd'] = consumption_data['meas_dd'].astype(str).str.zfill(2)  # Ensure day is two digits
consumption_data['date'] = consumption_data['meas_ym'] + '/' + consumption_data['meas_dd']


# Now I will drop the original meas_ym and meas_dd columns
consumption_data.drop(columns=['meas_ym', 'meas_dd'], inplace=True)


#Before continuing some supplies are known problematics, so drop them




# print(consumption_data.head())
#    Supply_ID  id val magnitude        date
# 0  SUPPLY001   1   0        A1  2022/10/20
# 1  SUPPLY001   2   0        A1  2022/10/20
# 2  SUPPLY001   3   0        A1  2022/10/20
# 3  SUPPLY001   4   0        A1  2022/10/20
# 4  SUPPLY001   5   0        A1  2022/10/20


#Now I will divide this data into different dataframes, one for each supply ID, so that when I run my model, I can use each supply ID as a separate input and don't have problem about the different number of days logged per supply. 

supply_ids = consumption_data['Supply_ID'].unique()
supply_dataframes = {}

for supply_id in supply_ids:
    supply_data = consumption_data[consumption_data['Supply_ID'] == supply_id].copy()
    supply_data.drop(columns=['Supply_ID'], inplace=True)  # Drop Supply_ID column
    supply_dataframes[supply_id] = supply_data

#Now that I have the dataframes for each supply ID, I can pivot the data so that each date is a row and each val is a column. Since we have 96 time points by day, I should expect 96 columns for each date. 

for supply_id, df in supply_dataframes.items():
    df_pivoted = df.pivot(index='date', columns='id', values='val')
    supply_dataframes[supply_id] = df_pivoted


#Now we will save this into a folder called "pivoted_data" in the current directory, with the name of the supply ID as the file name. 

import os
output_dir = "Study_datasets/Consumption/pivoted_data"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for supply_id, df in supply_dataframes.items():
    file_path = os.path.join(output_dir, f"{supply_id}.csv")
    df.to_csv(file_path, index=True)  # Save with date as index
    print(f"Saved {supply_id} data to {file_path}")
# Now we have the data saved in the pivoted_data folder, with each supply ID as a separate file.





