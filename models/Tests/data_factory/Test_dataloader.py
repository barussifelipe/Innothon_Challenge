import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from DataLoader import EnelSegLoader, get_loader_segment # Ensure this path is correct


# --- Configuration for testing ---
# (These should match the constants in your EnelSegLoader file)
PROCESSED_DATA_PATH = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Tests/Data/1825_days.csv'
DEFAULT_WIN_SIZE = 96
DEFAULT_STEP = 1
BATCH_SIZE = 4 # Use a small batch size for easier inspection

# Define the number of statistics you've added (e.g., mean, std, median = 3)
# Make sure this matches your implementation in EnelSegLoader's __getitem__
NUM_ADDED_STATS = 3 

print("--- Testing EnelSegLoader ---")

# --- Initialize DataLoaders ONCE for each mode ---
# The actual loading and processing happens inside EnelSegLoader's __init__
print(f"\nInitializing DataLoader for 'train' mode with data from: {PROCESSED_DATA_PATH}")
train_dataloader = get_loader_segment(
    data_path=PROCESSED_DATA_PATH,
    batch_size=BATCH_SIZE,
    win_size=DEFAULT_WIN_SIZE,
    step=DEFAULT_STEP,
    mode='train',
    dataset='Enel'
)
print("Train DataLoader initialized.")
print(f"Number of training sequences: {len(train_dataloader.dataset)}")


print(f"\nInitializing DataLoader for 'test' mode with data from: {PROCESSED_DATA_PATH}")
test_dataloader = get_loader_segment(
    data_path=PROCESSED_DATA_PATH,
    batch_size=BATCH_SIZE,
    win_size=DEFAULT_WIN_SIZE,
    step=DEFAULT_STEP,
    mode='test',
    dataset='Enel'
)
print("Test DataLoader initialized.")
print(f"Number of test sequences: {len(test_dataloader.dataset)}")
print(f"Number of validation sequences: {len(test_dataloader.dataset.val_sequences)}") # Accessing val_sequences via test_dataloader.dataset


# --- Test 1: Check DataLoader Iteration (Training Mode) ---
print("\n--- Checking DataLoader for Training Mode ---")
num_train_batches_to_check = min(3, len(train_dataloader)) 
if num_train_batches_to_check == 0:
    print("No training batches to check. Training set might be empty.")
else:
    for i, (sequences, labels) in enumerate(train_dataloader):
        print(f"\nTraining Batch {i+1}/{num_train_batches_to_check}:")
        # --- UPDATED CHECK ---
        expected_feature_dim = 1 + NUM_ADDED_STATS
        print(f"Sequences shape: {sequences.shape}") # Expected: [BATCH_SIZE, WIN_SIZE, 1 + NUM_ADDED_STATS]
        if sequences.shape[2] != expected_feature_dim:
            print(f"ERROR: Expected feature dimension {expected_feature_dim}, but got {sequences.shape[2]} for sequences.")
        # --- END UPDATED CHECK ---
        print(f"Labels shape: {labels.shape}")     # Expected: [BATCH_SIZE, WIN_SIZE]
        
        # Print first sequence's consumption and added stats
        print(f"Consumption (first sequence): \n{sequences[0, :, 0].numpy()}") 
        if NUM_ADDED_STATS > 0:
            print(f"Added Statistics (first sequence, first point): {sequences[0, 0, 1:].numpy()}") # Check first point's stats
            # Verify that stats are constant across the window for a single sequence
            if not np.all(sequences[0, :, 1:].numpy() == sequences[0, 0, 1:].numpy()):
                print("WARNING: Added statistics are NOT constant across the window for the first sequence.")

        print(f"Labels (first sequence): \n{labels[0, :].numpy()}")

        # Verify: labels should be all zeros for training data (only normal supplies)
        if not (labels == 0).all():
            print("WARNING: Training labels are NOT all zeros. This indicates non-regular data in training set.")

        if i + 1 >= num_train_batches_to_check:
            break

# --- Test 2: Check DataLoader Iteration (Test Mode) ---
print("\n--- Checking DataLoader for Test Mode ---")
num_test_batches_to_check = min(3, len(test_dataloader)) 
if num_test_batches_to_check == 0:
    print("No test batches to check. Test set might be empty.")
else:
    for i, (sequences, labels, metadata) in enumerate(test_dataloader):
        print(f"\nTest Batch {i+1}/{num_test_batches_to_check}:")

        expected_feature_dim = 1 + NUM_ADDED_STATS
        print(f"Sequences shape: {sequences.shape}") # Expected: [BATCH_SIZE, WIN_SIZE, 1 + NUM_ADDED_STATS]
        if sequences.shape[2] != expected_feature_dim:
            print(f"ERROR: Expected feature dimension {expected_feature_dim}, but got {sequences.shape[2]} for sequences.")
        print(f"Labels shape: {labels.shape}")     # Expected: [BATCH_SIZE, WIN_SIZE]
        print(f"Metadata type: {type(metadata)}") # Expected: dict
        print(f"Sample Metadata (first item in batch):")
        
        # Print first sequence's consumption and added stats
        print(f"Consumption (first sequence): \n{sequences[0, :, 0].numpy()}")
        if NUM_ADDED_STATS > 0:
            print(f"Added Statistics (first sequence, first point): {sequences[0, 0, 1:].numpy()}")
            if not np.all(sequences[0, :, 1:].numpy() == sequences[0, 0, 1:].numpy()):
                print("WARNING: Added statistics are NOT constant across the window for the first sequence.")

        # Metadata is a dict of lists/tensors from DataLoader's default_collate
        for key, value in metadata.items():
            print(f"  {key}: {value[0]}") # Print first item of each metadata field

        # Verify: check if labels for a non-regular supply are all 1s
        for j in range(sequences.shape[0]): # Iterate through items in the batch
            supply_id = metadata['Supply_ID'][j]
            is_non_regular_supply = metadata['Is_Non_Regular_Supply'][j].item() 

            print(f"  Sequence {j+1} (Supply: {supply_id}, Is_Non_Regular_Supply: {is_non_regular_supply}):")

            if is_non_regular_supply == 1:
                if not (labels[j, :] == 1).all():
                    print(f"    WARNING: Labels for non-regular supply {supply_id} are NOT all ones!")
            else:
                if not (labels[j, :] == 0).all():
                    print(f"    WARNING: Labels for regular supply {supply_id} are NOT all zeros!")

        if i + 1 >= num_test_batches_to_check:
            break

print("\n--- DataLoader testing complete ---")