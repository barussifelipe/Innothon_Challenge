import pandas as pd 
import numpy as np 

df_state = pd.read_csv("Study_datasets\Costumer_State\EsempioDataset - PAROLE DI STATO.csv", encoding="utf-16", sep="\t", decimal=",")

def State_Data_Treatment(df_state):

    #We will follow similar steps as in the Work_Data_Treatment function: 

    #We need to check if we have NA values in this dataset, and if so, in which columns. 
    df_state_na = df_state.isna().sum()
    # Supply_ID      0
    # meas_ts        0
    # ListaParole    1

    #We have 1 NA value in the column ListaParole, we will drop this row.
    df_state.dropna(inplace=True)

    #We will convert the column meas_ts to datetime format and extract the year.
    df_state["meas_ts"] = pd.to_datetime(df_state["meas_ts"]).dt.year

    #We will rename the column meas_ts to Year
    df_state.rename(columns={"meas_ts": "Year"}, inplace=True)

    #We will first understand the column ListaParole, we will check the unique values in this column.


    #We know that we have a list with a lot of unique values, so we need to check how many unique values we have in these permutations, we will use the explode function to do this. However, we need to convert the column to a list first. 

    #We will convert the column ListaParole to a list of strings, we will use the split function to do this.
    df_state["ListaParole"] = df_state["ListaParole"].apply(lambda x: x.split(","))

    #Now we will use the explode function to create a new row for each value in the list. 
    df_state_exploded = df_state.explode("ListaParole") 

    #We will check the unique values in the column ListaParole
    mappings = dict(enumerate(df_state_exploded["ListaParole"].unique()))
    # {0: 'PUP', 1: 'SCE1', 2: 'SHUNT', 3: 'WRNTHD', 4: 'CUT_OPEN_P', 5: 'BAT_LOW', 6: 'Reserved', 7: 'BPS_LOW', 8: 'FVP', 9: 'ALL', 10: 'CUWVO', 11: 'SGR', 12: 'POSDET', 13: 'FB_ERROR', 14: 'ACC', 15: 'DELTA_MEAS', 16: 'MET_OPEN', 17: 'ELPE_POLL', 18: 'PAD', 19: 'SCE2', 20: 'CPF3', 21: 'CPF2', 22: 'CPF1', 23: 'Unlocked ', 24: 'INT_BUF_FULL', 25: 'ORD', 26: 'CAPE', 27: 'INTA ', 28: 'PIC', 29: 'RF', 30: 'DDSP ', 31: 'WDOG ', 32: 'MCU:POW', 33: 'AFC'}

    #Now we will factorize the column ListaParole to convert it into numerical values. 
    df_state_exploded["ListaParole"] = pd.factorize(df_state_exploded["ListaParole"])[0]

    #Now we will aggregate the data to be one row per supply_id per year. 
    df_state_partial_agg = df_state.groupby(["Supply_ID", "Year"]).agg(
        Number_of_States=("Year", "count"),
    ).reset_index()

    number_of_states = df_state_partial_agg["Number_of_States"]

    df_state_agg = df_state_exploded.groupby(["Supply_ID", "Year"]).agg(
        Majority_Type_of_State=("ListaParole", lambda x: x.mode()[0]),
    ).reset_index()

    df_state_agg["Number_of_States"] = number_of_states


    return df_state_agg, mappings

if __name__ == "__main__":
    df_state_agg, mappings = State_Data_Treatment(df_state)
    #Save the dataframe to a csv file 
    df_state_agg.to_csv("Study_datasets\Costumer_State\State_processed.csv", index=False)

    








