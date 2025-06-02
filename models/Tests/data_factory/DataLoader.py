import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # Keep import for general utility, but won't be used in this loader
from sklearn.model_selection import train_test_split
# No longer needed: collections, numbers, math, PIL, pickle, id_to_timedelta, fill_missing_timestamps_for_supply

# --- Configuration ---
# Define the sequence length for the Anomaly Transformer
DEFAULT_WIN_SIZE = 96 # Example: 1 day of quarter-hourly data
DEFAULT_STEP = 1 # How much the window slides each time (1 for dense coverage)

# Supply-level train/test split ratio
TEST_SIZE_SUPPLIES = 0.2
RANDOM_STATE = 42

# --- Helper Function: Create Sequences for Multiple Supplies ---
def create_sequences_from_df(df, win_size, step):
    all_sequences = []
    all_labels = []
    all_metadata = []

    print(f"Starting sequence creation for a DataFrame with {len(df)} rows...")
    for supply_id, supply_data in df.groupby('Supply_ID'):
        scaled_values = supply_data['Scaled_Consumption'].values
        timestamps = supply_data['Timestamp'].values # Get as NumPy array

        is_non_regular_supply = supply_data['Is_Non_Regular'].iloc[0]

        if len(scaled_values) < win_size:
            continue

        for i in range(0, len(scaled_values) - win_size + 1, step):
            sequence = scaled_values[i : i + win_size]
            label_array = np.full(win_size, is_non_regular_supply, dtype=np.float32)

            all_sequences.append(np.float32(sequence))
            all_labels.append(label_array)
            
            # --- MODIFICATION HERE: Convert numpy.datetime64 to Unix timestamp (int seconds) ---
            start_unix_timestamp = timestamps[i].astype(np.int64) // 10**9
            end_unix_timestamp = timestamps[i + win_size - 1].astype(np.int64) // 10**9

            all_metadata.append({
                'Supply_ID': supply_id,
                'Start_Timestamp': start_unix_timestamp,
                'End_Timestamp': end_unix_timestamp,
                'Is_Non_Regular_Supply': is_non_regular_supply
            })
    print(f"Finished sequence creation. Total sequences: {len(all_sequences)}")
    return all_sequences, all_labels, all_metadata

# --- Adapted EnelSegLoader Class ---
class EnelSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        # self.scaler = StandardScaler() # No longer needed if data is pre-scaled

        # --- 1. Load Processed Data ---
        # data_path is now the path to the single processed CSV file
        df_consumption = pd.read_csv(data_path)

        # Drop the unnamed index column if it exists from previous save
        if 'Unnamed: 0' in df_consumption.columns:
            df_consumption = df_consumption.drop(columns=['Unnamed: 0'])
        
        # Convert Timestamp column to datetime objects
        df_consumption['Timestamp'] = pd.to_datetime(df_consumption['Timestamp'])

        print("Sorting entire DataFrame by Supply_ID and Timestamp (one time)...")
        df_consumption = df_consumption.sort_values(by=['Supply_ID', 'Timestamp']).reset_index(drop=True)
        print("Sorting complete.")

        # Assuming 'val' column is ALREADY scaled and is the feature we need
        df_consumption['Scaled_Consumption'] = df_consumption['val']
        print("Data is assumed to be pre-scaled. Using 'val' column as 'Scaled_Consumption'.")
        
        #Computing statistics
        print("Calculating global statistics for each Supply_ID...")
        self.supply_statistics = {}
        for supply_id, group_df in df_consumption.groupby('Supply_ID'):
            mean_val = group_df['Scaled_Consumption'].mean()
            std_val = group_df['Scaled_Consumption'].std()
            median_val = group_df['Scaled_Consumption'].median()
            
            # Handle cases where std might be NaN (e.g., supply with only one reading or constant values)
            if pd.isna(std_val):
                std_val = 0.0 # Assign a default, or a very small epsilon if division by std is later expected
            
            self.supply_statistics[supply_id] = {
                'mean_consumption': mean_val,
                'std_consumption': std_val,
                'median_consumption': median_val,
            }
        print(f"Finished calculating statistics for {len(self.supply_statistics)} supplies.")


        # --- 2. Perform Supply-Level Split ---
        print('Performing Supply-Level Split ')
        supply_labels = df_consumption[['Supply_ID', 'Is_Non_Regular']].drop_duplicates()

        train_supply_ids, test_supply_ids = train_test_split(
            supply_labels['Supply_ID'],
            test_size=TEST_SIZE_SUPPLIES,
            stratify=supply_labels['Is_Non_Regular'],
            random_state=RANDOM_STATE
        )
        print('Supply-Level Split  done')
        # --- 3. Filter dataframes based on the split Supply_IDs ---
        print('Filtering dataframe (only regular for training)')
        # Train data: Only sequences from regular supplies in the training split
        # We need the full df_consumption to filter by Supply_ID
        df_train_filtered = df_consumption[
            df_consumption['Supply_ID'].isin(train_supply_ids) &
            (df_consumption['Is_Non_Regular'] == 0) # Only normal data for training
        ].copy()

        # Test data: Sequences from all supplies in the test split
        df_test_filtered = df_consumption[
            df_consumption['Supply_ID'].isin(test_supply_ids)
        ].copy()

        # Check if training data for normal supplies is empty
        if df_train_filtered.empty:
            raise ValueError("No 'Regolare' supplies found in the training split. Cannot create training sequences.")

        print('Filtering done')

        # --- 4. Generate Sequences for different modes ---
        print('Begin generating sequences')

        #Train sequences
        self.train_sequences, self.train_labels, self.train_metadata = create_sequences_from_df(
            df_train_filtered, self.win_size, self.step
        )
        
        #Test sequences
        self.test_sequences, self.test_labels, self.test_metadata = create_sequences_from_df(
            df_test_filtered, self.win_size, self.step
        )
        
        # Validation data: For simplicity, use test data for val
        # In a real scenario, you might split a portion of your training set for validation
        self.val_sequences = self.test_sequences
        self.val_labels = self.test_labels
        self.val_metadata = self.test_metadata

        # The thresholding data should be entirely normal. Using the training sequences
        # ensures this, as they were filtered to be non-regular (normal) supplies.
        self.thre_sequences = self.train_sequences
        self.thre_labels = self.train_labels # These labels will all be 0 (normal)
        self.thre_metadata = self.train_metadata # Metadata reflects normal too
        

        print(f"Train sequences count: {len(self.train_sequences)}")
        print(f"Test sequences count: {len(self.test_sequences)}")
        print(f"Validation sequences count: {len(self.val_sequences)}")
        print(f"Thresholding sequences count: {len(self.thre_sequences)}") # Optional: Add this print for clarity





    def __len__(self):
        if self.mode == "train":
            return len(self.train_sequences)
        elif self.mode == 'val':
            return len(self.val_sequences)
        elif self.mode == 'test':
            return len(self.test_sequences)
        else: # Default case, e.g., for custom inference modes
            return len(self.test_sequences) # Fallback to test length

    def __getitem__(self, index):
        if self.mode == "train":
            current_sequences = self.train_sequences
            current_labels = self.train_labels
            current_metadata = self.train_metadata
        elif self.mode == 'val':
            current_sequences = self.val_sequences
            current_labels = self.val_labels
            current_metadata = self.val_metadata
        elif self.mode == 'test':
            current_sequences = self.test_sequences
            current_labels = self.test_labels
            current_metadata = self.test_metadata
        elif self.mode == 'thre': # --- ADD THIS CONDITION ---
            current_sequences = self.thre_sequences
            current_labels = self.thre_labels
            current_metadata = self.thre_metadata
        else: # Fallback for any other custom modes you might add later
            print(f"Warning: Unknown mode '{self.mode}'. Falling back to test sequences.")
            current_sequences = self.test_sequences
            current_labels = self.test_labels
            current_metadata = self.test_metadata
        # Get the consumption sequence for the current index
        sequence_data = current_sequences[index] # This is a numpy array, shape (win_size,)

        # Get the metadata for the current sequence to extract the Supply_ID
        metadata_item = current_metadata[index]
        supply_id = metadata_item['Supply_ID']

        # Retrieve the pre-calculated supply statistics
        supply_stats = self.supply_statistics.get(supply_id)

        # Define default statistics for safety, or raise an error if stats are mandatory
        if supply_stats is None:
            # You might want to log a warning here if this happens frequently
            print(f"Warning: No statistics found for Supply_ID: {supply_id}. Using default zeros.")
            mean_c, std_c, median_c = 0.0, 0.0, 0.0 # Or global average, or raise an error
        else:
            mean_c = supply_stats['mean_consumption']
            std_c = supply_stats['std_consumption']
            median_c = supply_stats['median_consumption']

        # Create a numpy array of these static features for the current supply
        # The order here should match what your model expects
        static_features_for_window = np.array([mean_c, std_c, median_c], dtype=np.float32) # Shape (num_stats,)

        # Repeat the static features for each point in the window (win_size times)
        # This transforms (num_stats,) into (win_size, num_stats)
        repeated_static_features = np.tile(static_features_for_window, (self.win_size, 1))

        # Combine the consumption sequence with the repeated static features
        # Reshape sequence_data from (win_size,) to (win_size, 1) to allow concatenation along axis=1
        combined_sequence = np.concatenate(
            [sequence_data.reshape(-1, 1), repeated_static_features],
            axis=1 # Concatenate along the feature dimension
        ) # Resulting shape: (self.win_size, 1 + num_stats)

        # Convert the combined sequence and labels to PyTorch tensors
        sequences_tensor = torch.tensor(combined_sequence, dtype=torch.float32)
        labels_tensor = torch.tensor(current_labels[index], dtype=torch.float32)

        # Return values based on the mode
        if self.mode == 'test':
            return sequences_tensor, labels_tensor, metadata_item
        else:
            return sequences_tensor, labels_tensor


# --- Modified get_loader_segment function ---
def get_loader_segment(data_path, batch_size, win_size=DEFAULT_WIN_SIZE, step=DEFAULT_STEP, mode='train', dataset='Enel'):
    """
    Returns a DataLoader for the specified dataset and mode.
    """
    if dataset == 'Enel':
        dataset_loader = EnelSegLoader(data_path, win_size, step, mode)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Please choose 'Enel'.")

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset_loader,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0, # Set to 0 for simpler debugging, increase for performance
                             drop_last=True if mode == 'train' else False # Drop last batch in train for consistent size
                            )
    return data_loader