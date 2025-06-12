import pandas as pd
import os

#Paths
DATASETS_PATHS = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Study_datasets/Consumption/pivoted_data'

#File names
file_names = os.listdir(DATASETS_PATHS)

#Check for NaN
'''
As for the format I want the schema (Supply_ID, Day, Quarter1, Quarter2... Quarter96), that will be fed to the data Loader
'''


print(file_names)

