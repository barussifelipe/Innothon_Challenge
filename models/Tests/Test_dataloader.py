import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from DataLoader import EnelSegLoader, get_loader_segment # Ensure this path is correct


# --- Configuration for testing ---
# (These should match the constants in your EnelSegLoader file)
PROCESSED_DATA_PATH = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Tests/Data/1300.csv'
DEFAULT_WIN_SIZE = 96
DEFAULT_STEP = 1
BATCH_SIZE = 4 # Use a small batch size for easier inspection

print("--- Testing EnelSegLoader ---")

# --- Initialize DataLoaders ONCE for each mode ---
# The actual loading and processing happens inside EnelSegLoader's __init__
# This will trigger the full __init__ process (sorting, sequence generation) once for train mode
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


# This will trigger the full __init__ process (sorting, sequence generation) once for test mode
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
num_train_batches_to_check = min(3, len(train_dataloader)) # Check first 3 batches or fewer if less available
if num_train_batches_to_check == 0:
    print("No training batches to check. Training set might be empty.")
else:
    for i, (sequences, labels) in enumerate(train_dataloader):
        print(f"\nTraining Batch {i+1}/{num_train_batches_to_check}:")
        print(f"Sequences shape: {sequences.shape}") # Expected: [BATCH_SIZE, WIN_SIZE, 1]
        print(f"Labels shape: {labels.shape}")     # Expected: [BATCH_SIZE, WIN_SIZE]
        print(f"Sequences (first): \n{sequences[0, :, 0].numpy()}") # Print first sequence in batch
        print(f"Labels (first): \n{labels[0, :].numpy()}") # Print labels for first sequence

        # Verify: labels should be all zeros for training data (only normal supplies)
        if not (labels == 0).all():
            print("WARNING: Training labels are NOT all zeros. This indicates non-regular data in training set.")

        if i + 1 >= num_train_batches_to_check:
            break

# --- Test 2: Check DataLoader Iteration (Test Mode) ---
print("\n--- Checking DataLoader for Test Mode ---")
num_test_batches_to_check = min(3, len(test_dataloader)) # Check first 3 batches
if num_test_batches_to_check == 0:
    print("No test batches to check. Test set might be empty.")
else:
    for i, (sequences, labels, metadata) in enumerate(test_dataloader):
        print(f"\nTest Batch {i+1}/{num_test_batches_to_check}:")
        print(f"Sequences shape: {sequences.shape}") # Expected: [BATCH_SIZE, WIN_SIZE, 1]
        print(f"Labels shape: {labels.shape}")     # Expected: [BATCH_SIZE, WIN_SIZE]
        print(f"Metadata type: {type(metadata)}") # Expected: dict
        print(f"Sample Metadata (first item in batch):")
        # Metadata is a dict of lists/tensors from DataLoader's default_collate
        for key, value in metadata.items():
            # For each key, value is a list of batch_size items or a tensor
            print(f"  {key}: {value[0]}") # Print first item of each metadata field

        # Verify: check if labels for a non-regular supply are all 1s
        for j in range(sequences.shape[0]): # Iterate through items in the batch
            # metadata['Supply_ID'] is now a list/tensor, access with [j]
            supply_id = metadata['Supply_ID'][j]
            is_non_regular_supply = metadata['Is_Non_Regular_Supply'][j].item() # .item() for scalar tensor if it's a tensor

            print(f"  Sequence {j+1} (Supply: {supply_id}, Is_Non_Regular_Supply: {is_non_regular_supply}):")
            # print(f"    Labels: {labels[j, :].numpy()}") # Uncomment to see full labels

            if is_non_regular_supply == 1:
                if not (labels[j, :] == 1).all():
                    print(f"    WARNING: Labels for non-regular supply {supply_id} are NOT all ones!")
            else:
                if not (labels[j, :] == 0).all():
                    print(f"    WARNING: Labels for regular supply {supply_id} are NOT all zeros!")

        if i + 1 >= num_test_batches_to_check:
            break

print("\n--- DataLoader testing complete ---")