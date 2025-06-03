import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def align_anomaly_scores_with_dates(
    original_data_dir: str,
    train_scores_dir: str,
    test_scores_dir: str,
    test_size_split: float = 0.2, # MUST match EnergySegLoader's test_size
    random_state_split: int = 42, # MUST match EnergySegLoader's random_state
    output_dir: str = "aligned_anomaly_data"
):
    """
    Aligns anomaly scores (train and test) with their original dates and
    saves a combined CSV for each supply, correctly handling dropped NaN rows.

    Args:
        original_data_dir (str): Directory containing the original supply CSV files
                                 (e.g., 'SUPPLY001.csv', 'SUPPLY009.csv').
        train_scores_dir (str): Directory containing saved train anomaly score .npy files.
        test_scores_dir (str): Directory containing saved raw test anomaly score .npy files.
        test_size_split (float): The test_size used in train_test_split in EnergySegLoader.
        random_state_split (int): The random_state used in train_test_split in EnergySegLoader.
        output_dir (str): Directory to save the combined aligned CSV files.
    """

    os.makedirs(output_dir, exist_ok=True)
    print(f"Aligned data will be saved to: {os.path.abspath(output_dir)}")

    processed_supplies_count = 0

    # Get a list of all supply IDs from the original data directory
    supply_files = [f for f in os.listdir(original_data_dir) if f.endswith('.csv')]
    supply_ids = [f.replace('.csv', '') for f in supply_files]

    print("\n--- Starting alignment process for all supplies ---")

    for supply_id in sorted(supply_ids): # Sort for consistent processing order
        print(f"\nProcessing supply: {supply_id}")

        original_filepath = os.path.join(original_data_dir, f"{supply_id}.csv")
        train_scores_filepath = os.path.join(train_scores_dir, f"{supply_id}_train_anomaly_scores.npy")
        test_scores_filepath = os.path.join(test_scores_dir, f"{supply_id}_raw_test_anomaly_scores.npy")

        # --- 1. Load Original Data with Dates ---
        try:
            full_original_df = pd.read_csv(original_filepath)
            full_original_df['date'] = pd.to_datetime(full_original_df['date'])
            
            # Convert consumption columns to float, handling commas
            # This is crucial as the DataLoader expects float data
            consumption_cols = full_original_df.columns.drop('date')
            for col in consumption_cols:
                full_original_df[col] = full_original_df[col].astype(str).str.replace(',', '.').astype(float)
            
            print(f"  Loaded original data ({len(full_original_df)} days).")
        except FileNotFoundError:
            print(f"  Error: Original data file not found for {supply_id}. Skipping.")
            continue
        except Exception as e:
            print(f"  Error loading original data for {supply_id}: {e}. Skipping.")
            continue

        # --- 2. Replicate DataLoader's NaN Handling and Train/Test Split ---
        # IMPORTANT: This section mirrors the DataLoader's __init__ method
        
        # Drop the 'date' column for the split, just like in DataLoader
        data_for_split = full_original_df.drop(columns=["date"])
        
        # Apply dropna() BEFORE the split, just like in DataLoader
        initial_rows = len(data_for_split)
        data_for_split = data_for_split.dropna()
        dropped_rows = initial_rows - len(data_for_split)
        if dropped_rows > 0:
            print(f"  Dropped {dropped_rows} rows with NaN values for {supply_id} (matching DataLoader).")
        
        # Perform the train/test split on the cleaned data
        data_train_cleaned, data_test_cleaned = train_test_split(
            data_for_split, test_size=test_size_split, random_state=random_state_split
        )

        # Now, get the corresponding dates for the cleaned and split data
        # This requires re-indexing the original full_original_df based on the indices
        # that survived the dropna and split.
        
        # Get the indices of the rows that survived the dropna
        survived_indices = data_for_split.index
        
        # Filter the full_original_df to only include rows that survived dropna
        filtered_original_df = full_original_df.loc[survived_indices].copy()

        # Perform the train_test_split on the filtered_original_df to get aligned dates
        # This split will use the same random_state and test_size, ensuring the dates
        # match the data_train_cleaned and data_test_cleaned
        train_dates_df, test_dates_df = train_test_split(
            filtered_original_df[['date']], test_size=test_size_split, random_state=random_state_split
        )
        
        # Sort both split dataframes by date to ensure chronological order.
        # The anomaly scores are generated sequentially based on the order the data is loaded.
        train_dates_df = train_dates_df.sort_values(by='date').reset_index(drop=True)
        test_dates_df = test_dates_df.sort_values(by='date').reset_index(drop=True)

        print(f"  Replicated split: Train days {len(train_dates_df)}, Test days {len(test_dates_df)}.")

        # --- 3. Load Anomaly Scores ---
        try:
            train_scores = np.load(train_scores_filepath)
            if train_scores.ndim > 1: train_scores = train_scores.flatten()
            print(f"  Loaded train anomaly scores ({len(train_scores)} scores).")

            test_scores = np.load(test_scores_filepath)
            if test_scores.ndim > 1: test_scores = test_scores.flatten()
            print(f"  Loaded test anomaly scores ({len(test_scores)} scores).")
        except FileNotFoundError:
            print(f"  Error: Anomaly score .npy files not found for {supply_id}. Skipping.")
            continue
        except Exception as e:
            print(f"  Error loading anomaly scores for {supply_id}: {e}. Skipping.")
            continue

        # --- 4. Verify Length Alignment ---
        if len(train_scores) != len(train_dates_df):
            print(f"  CRITICAL ERROR: Train scores length ({len(train_scores)}) does not match "
                  f"aligned train data length ({len(train_dates_df)}) for {supply_id}. Skipping.")
            continue
        if len(test_scores) != len(test_dates_df):
            print(f"  CRITICAL ERROR: Test scores length ({len(test_scores)}) does not match "
                  f"aligned test data length ({len(test_dates_df)}) for {supply_id}. Skipping.")
            continue

        # --- 5. Create Combined DataFrames ---
        train_combined_df = pd.DataFrame({
            'date': train_dates_df['date'],
            'anomaly_score': train_scores,
            'is_test_day': 0 # 0 for train day
        })

        test_combined_df = pd.DataFrame({
            'date': test_dates_df['date'],
            'anomaly_score': test_scores,
            'is_test_day': 1 # 1 for test day
        })

        # --- 6. Concatenate, Sort, and Save ---
        final_combined_df = pd.concat([train_combined_df, test_combined_df])
        final_combined_df = final_combined_df.sort_values(by='date').reset_index(drop=True)

        output_filepath = os.path.join(output_dir, f"{supply_id}_aligned_anomaly_data.csv")
        final_combined_df.to_csv(output_filepath, index=False)
        print(f"  Aligned data saved for {supply_id} to '{output_filepath}'.")
        processed_supplies_count += 1

    print(f"\n--- Alignment process complete. Successfully processed {processed_supplies_count} supplies. ---")


if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Adjust these paths and parameters for your setup

    # Directory where your original supply CSV files are located
    ORIGINAL_DATA_DIRECTORY = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Study_datasets/Consumption/pivoted_data'
    
    # Directory where train anomaly scores are saved by solver.py
    TRAIN_SCORES_DIRECTORY = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Setup_Anomaly/src/Anomaly_Transformer/checkpoints/train_anomaly_scores_for_global_threshold'
    
    # Directory where raw test anomaly scores are saved by solver.py
    TEST_SCORES_DIRECTORY = '/Users/diegozago2312/Documents/Innothon/Challenge2/Setup_Anomaly/src/Anomaly_Transformer/checkpoints/raw_test_anomaly_scores_for_application'
    
    # These MUST match the test_size and random_state used in your EnergySegLoader's train_test_split
    # and in your main.py/solver.py setup. Consistency is key for correct alignment.
    TEST_SPLIT_SIZE_USED = 0.2
    RANDOM_SPLIT_STATE_USED = 42

    # Run the alignment process
    align_anomaly_scores_with_dates(
        original_data_dir=ORIGINAL_DATA_DIRECTORY,
        train_scores_dir=TRAIN_SCORES_DIRECTORY,
        test_scores_dir=TEST_SCORES_DIRECTORY,
        test_size_split=TEST_SPLIT_SIZE_USED,
        random_state_split=RANDOM_SPLIT_STATE_USED
    )