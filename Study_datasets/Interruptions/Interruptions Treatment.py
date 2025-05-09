import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

#Load the INTERUPTIONS dataset with specific encoding, separator and decimal format.
df_interruptions = pd.read_csv("Study_datasets\Interruptions\EsempioDataset - INTERRUZIONI.csv", encoding="utf-16", sep="\t", decimal=",")

def Interruptions_Data_Treatment(df_interruptions): 
    #We will follow similar steps as in the Work_Data_Treatment function:
    #We know by the analysis that we don't have NA values in this dataset. 

    #We will convert the column start_date and end_date to datetime format and extract the year. 
    df_interruptions["start_date"] = pd.to_datetime(df_interruptions["start_date"]).dt.year
    df_interruptions["end_date"] = pd.to_datetime(df_interruptions["end_date"]).dt.year

    #First we need to check if one ends in a year and the other in a different year, seems improbable but we need to check.

    # Check if the values in 'start_date' and 'end_date' are different
    different_years = df_interruptions["start_date"] != df_interruptions["end_date"]

    # Print rows where the years are different
    # print(df_interruptions[different_years])
    # Index: []

    #No rows were printed, so we can proceed to the next step.

    #Rename the columns start_date to Year
    df_interruptions.rename(columns={"start_date": "Year"}, inplace=True)

    #We drop the column end_date as it is not needed anymore.
    df_interruptions.drop(columns=["end_date"], inplace=True) 

    #We will save the mapping of the factorized column for later use
    mappings = dict(enumerate(df_interruptions["tipologia_interruzione"].unique()))

    #We will factorize the column tipologia_interruzione to convert it into numerical values 
    df_interruptions["tipologia_interruzione"] = pd.factorize(df_interruptions["tipologia_interruzione"])[0]


    #Now we will aggregate the data to be one row per supply_id per year. 

    df_interruptions_agg = df_interruptions.groupby(["Supply_ID", "Year"]).agg(
        Number_of_Interruptions=("Year", "count"),
        Majority_Type_of_Interruption=("tipologia_interruzione", lambda x: x.mode()[0]),
        Max=("durata_netta", "max"),
        Min=("durata_netta", "min"),
        Mean=("durata_netta", lambda x: round(x.mean(), 2)),
        Std=("durata_netta", lambda x: round(x.std(), 2) if len(x) > 1 else 0),
    ).reset_index()

    return df_interruptions_agg, mappings

if __name__ == "__main__":
    df_interruptions_agg, mappings = Interruptions_Data_Treatment(df_interruptions)
    #Save the dataframe to a csv file 
    df_interruptions_agg.to_csv("Interruptions_processed.csv", index=False)

    




