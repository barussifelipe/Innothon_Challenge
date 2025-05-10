import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

#Load the LAVORI dataset with specific encoding, separator and decimal format.
df_lavori = pd.read_csv("EsempioDataset - LAVORI.csv", encoding="utf-16", sep="\t", decimal=",")

def Work_Data_Treatment(df_lavori): 
    #Now, we will:

    #Drop NA values in the columns "woa_activity_type" and "woa_activity_subtype" 
    df_lavori.dropna(subset= ["woa_activity_type", "woa_activity_subtype"], inplace=True)
    #Convert the column "woe_dt_execution" to datetime format and extract the year
    df_lavori["woe_dt_execution"] = pd.to_datetime(df_lavori["woe_dt_execution"]).dt.year

    #Rename the column "woe_dt_execution" to "Year"
    df_lavori.rename(columns={"woe_dt_execution": "Year"}, inplace=True)
    #Getting the categorical columns to be factorized
    categorical_columns = ["woa_activity_type", "woa_activity_subtype"]

    #Save the mappings of the factorized columns for later use
    mappings = {col: dict(enumerate(df_lavori[col].unique())) for col in categorical_columns}

    #Factorize the columns "woa_activity_type" and "woa_activity_subtype" to convert them into numerical values

    for col in categorical_columns: 
        df_lavori[col] = pd.factorize(df_lavori[col])[0]


    #Now we will aggreate the data of supply_id to be one row per supply_id per year. 
    #We will count the number of works and find the majority type and subtype of work for each supply_id per year.
    df_lavori_agg = df_lavori.groupby(["Supply_ID", "Year"]).agg(
        Number_of_Works=("Year", "count"),
        Majority_Type_of_Work=("woa_activity_type", lambda x: x.mode()[0]),
        Majority_Sub_Type_of_Work=("woa_activity_subtype", lambda x: x.mode()[0])
    ).reset_index()

    return df_lavori_agg, mappings 

if __name__ == "__main__":
    df_lavori_agg, mappings = Work_Data_Treatment(df_lavori)
    #Save the dataframe to a csv file 
    df_lavori_agg.to_csv("Work_processed.csv", index=False)
    

