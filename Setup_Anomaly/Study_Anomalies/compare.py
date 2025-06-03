import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_prepare_data(anomaly_score_filepath: str, original_data_filepath: str, 
                          test_size: float, random_state: int, anormly_ratio: float):
    """
    Loads anomaly scores and corresponding original consumption data for a single supply.
    Calculates threshold and identifies anomalous days.

    Args:
        anomaly_score_filepath (str): Full path to the anomaly score CSV file.
        original_data_filepath (str): Full path to the original, unscaled consumption CSV.
        test_size (float): The test_size used in train_test_split for the original data.
        random_state (int): The random_state used in train_test_split for the original data.
        anormly_ratio (float): The anomaly ratio used for thresholding.

    Returns:
        tuple: (supply_id, scores, threshold, anomalous_days_binary, data_test_original)
               Returns None if an error occurs.
    """
    supply_id = os.path.basename(anomaly_score_filepath).replace("_anomaly_scores.csv", "")
    print(f"Loading data for supply: {supply_id}")

    try:
        # Load Anomaly Scores
        scores = pd.read_csv(anomaly_score_filepath, header=None).iloc[:, 0].values
        if len(scores) == 0:
            print(f"Error: Anomaly score file '{anomaly_score_filepath}' is empty. Skipping.")
            return None
        print(f"  Loaded {len(scores)} anomaly scores.")

        # Load Original Consumption Data and re-split
        full_original_data = pd.read_csv(original_data_filepath).drop(columns=["date"]).apply(lambda col: col.str.replace(',', '.').astype(float))
        
        # Re-create the train/test split exactly as in EnergySegLoader
        data_train_original, data_test_original = train_test_split(
            full_original_data, test_size=test_size, random_state=random_state
        )
        
        # Crucial: Ensure the test data length matches the anomaly scores length
        # This handles cases where the last batch might be smaller or data_loader pads
        if len(data_test_original) != len(scores):
            print(f"  Warning: Length mismatch for {supply_id}. Original test data ({len(data_test_original)} days) "
                  f"vs. anomaly scores ({len(scores)} scores). Adjusting original test data length.")
            data_test_original = data_test_original.iloc[:len(scores)]
        
        print(f"  Extracted test set consumption data for {len(data_test_original)} days.")

        # Calculate Threshold
        threshold = np.percentile(scores, 100 - anormly_ratio)
        
        # Identify Anomalous Days
        anomalous_days_binary = (scores > threshold).astype(int)
        
        return supply_id, scores, threshold, anomalous_days_binary, data_test_original

    except FileNotFoundError:
        print(f"Error: File not found for {supply_id} at '{anomaly_score_filepath}' or '{original_data_filepath}'.")
        return None
    except Exception as e:
        print(f"Error processing data for {supply_id}: {e}")
        return None


def compare_supply_insights(
    regular_supply_anomaly_file: str,
    regular_supply_original_data_file: str,
    non_regular_supply_anomaly_file: str,
    non_regular_supply_original_data_file: str,
    anormly_ratio: float,
    test_size_split: float = 0.2, # Must match data_loader's test_size
    random_state_split: int = 42, # Must match data_loader's random_state
    output_dir: str = "comparative_analysis_results"
):
    """
    Compares anomaly scores and consumption patterns between a regular and non-regular supply.

    Args:
        regular_supply_anomaly_file (str): Path to anomaly scores for the regular supply.
        regular_supply_original_data_file (str): Path to original data for the regular supply.
        non_regular_supply_anomaly_file (str): Path to anomaly scores for the non-regular supply.
        non_regular_supply_original_data_file (str): Path to original data for the non-regular supply.
        anormly_ratio (float): The anomaly ratio used for thresholding.
        test_size_split (float): The test_size used in train_test_split for the original data.
        random_state_split (int): The random_state used in train_test_split for the original data.
        output_dir (str): Directory to save analysis results (plots).
    """

    os.makedirs(output_dir, exist_ok=True)
    print(f"Comparative analysis results will be saved to: {os.path.abspath(output_dir)}")

    # --- 1. Load and Prepare Data for Both Supplies ---
    print("\n--- Loading and preparing data for both supplies ---")
    regular_data = load_and_prepare_data(
        regular_supply_anomaly_file, regular_supply_original_data_file, 
        test_size_split, random_state_split, anormly_ratio
    )
    non_regular_data = load_and_prepare_data(
        non_regular_supply_anomaly_file, non_regular_supply_original_data_file,
        test_size_split, random_state_split, anormly_ratio
    )

    if regular_data is None or non_regular_data is None:
        print("Failed to load data for one or both supplies. Exiting comparison.")
        return

    reg_id, reg_scores, reg_thresh, reg_anom_binary, reg_data_test = regular_data
    non_reg_id, non_reg_scores, non_reg_thresh, non_reg_anom_binary, non_reg_data_test = non_regular_data

    print(f"\nRegular Supply ({reg_id}): {np.sum(reg_anom_binary)} anomalous days ({np.sum(reg_anom_binary)/len(reg_scores)*100:.2f}%)")
    print(f"Non-Regular Supply ({non_reg_id}): {np.sum(non_reg_anom_binary)} anomalous days ({np.sum(non_reg_anom_binary)/len(non_reg_scores)*100:.2f}%)")

    # --- 2. Comparative Visualization: Anomaly Scores Over Time ---
    print("\n--- Generating Comparative Anomaly Score Plot ---")
    plt.figure(figsize=(18, 8))
    
    # Regular Supply
    plt.plot(np.arange(len(reg_scores)), reg_scores, label=f'{reg_id} Anomaly Score', color='blue', alpha=0.7)
    reg_anomalous_indices = np.where(reg_anom_binary == 1)[0]
    plt.scatter(reg_anomalous_indices, reg_scores[reg_anomalous_indices], 
                color='cyan', s=60, zorder=5, label=f'{reg_id} Anomalous Day')
    plt.axhline(y=reg_thresh, color='blue', linestyle='--', label=f'{reg_id} Threshold ({reg_thresh:.2f})')

    # Non-Regular Supply
    # Offset the non-regular supply's plot if lengths are very different, or plot on separate axes
    # For now, assuming similar lengths and plotting on same axis for direct comparison
    plt.plot(np.arange(len(non_reg_scores)), non_reg_scores, label=f'{non_reg_id} Anomaly Score', color='orange', alpha=0.7)
    non_reg_anomalous_indices = np.where(non_reg_anom_binary == 1)[0]
    plt.scatter(non_reg_anomalous_indices, non_reg_scores[non_reg_anomalous_indices], 
                color='red', s=60, zorder=5, label=f'{non_reg_id} Anomalous Day')
    plt.axhline(y=non_reg_thresh, color='orange', linestyle='--', label=f'{non_reg_id} Threshold ({non_reg_thresh:.2f})')

    plt.title(f'Comparative Anomaly Scores Over Time: {reg_id} vs. {non_reg_id}')
    plt.xlabel('Day Index in Test Set')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'comparative_anomaly_scores_{reg_id}_vs_{non_reg_id}.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Comparative anomaly score plot saved to '{plot_filename}'")

    # --- 3. Comparative Visualization: Histograms of Anomaly Scores ---
    print("\n--- Generating Comparative Anomaly Score Histograms ---")
    plt.figure(figsize=(16, 7))

    plt.subplot(1, 2, 1)
    sns.histplot(reg_scores, bins=50, kde=True, color='blue')
    plt.axvline(x=reg_thresh, color='green', linestyle='--', label=f'Threshold ({reg_thresh:.2f})')
    plt.title(f'Distribution of Anomaly Scores: {reg_id}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    sns.histplot(non_reg_scores, bins=50, kde=True, color='orange')
    plt.axvline(x=non_reg_thresh, color='red', linestyle='--', label=f'Threshold ({non_reg_thresh:.2f})')
    plt.title(f'Distribution of Anomaly Scores: {non_reg_id}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    hist_filename = os.path.join(output_dir, f'comparative_anomaly_histograms_{reg_id}_vs_{non_reg_id}.png')
    plt.savefig(hist_filename)
    plt.close()
    print(f"Comparative anomaly histograms saved to '{hist_filename}'")

    # --- 4. Comparative Visualization: Average Consumption Profiles ---
    print("\n--- Generating Comparative Average Consumption Profiles ---")

    # Regular Supply Average Profiles
    reg_normal_day_indices = np.where(reg_anom_binary == 0)[0]
    avg_reg_normal_profile = reg_data_test.iloc[reg_normal_day_indices].mean(axis=0) if len(reg_normal_day_indices) > 0 else np.zeros(reg_data_test.shape[1])
    avg_reg_anomalous_profile = reg_data_test.iloc[reg_anomalous_indices].mean(axis=0) if len(reg_anomalous_indices) > 0 else np.zeros(reg_data_test.shape[1])

    # Non-Regular Supply Average Profiles
    non_reg_normal_day_indices = np.where(non_reg_anom_binary == 0)[0]
    avg_non_reg_normal_profile = non_reg_data_test.iloc[non_reg_normal_day_indices].mean(axis=0) if len(non_reg_normal_day_indices) > 0 else np.zeros(non_reg_data_test.shape[1])
    avg_non_reg_anomalous_profile = non_reg_data_test.iloc[non_reg_anomalous_indices].mean(axis=0) if len(non_reg_anomalous_indices) > 0 else np.zeros(non_reg_data_test.shape[1])

    plt.figure(figsize=(18, 8))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, 97), avg_reg_normal_profile, label='Avg Normal Day', color='green', linestyle='-')
    if len(reg_anomalous_indices) > 0:
        plt.plot(np.arange(1, 97), avg_reg_anomalous_profile, label='Avg Anomalous Day', color='red', linestyle='--')
    plt.title(f'Avg Daily Consumption Profile: {reg_id}')
    plt.xlabel('Quarter-Hour of Day (1-96)')
    plt.ylabel('Average Consumption Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, 97), avg_non_reg_normal_profile, label='Avg Normal Day', color='green', linestyle='-')
    if len(non_reg_anomalous_indices) > 0:
        plt.plot(np.arange(1, 97), avg_non_reg_anomalous_profile, label='Avg Anomalous Day', color='red', linestyle='--')
    plt.title(f'Avg Daily Consumption Profile: {non_reg_id}')
    plt.xlabel('Quarter-Hour of Day (1-96)')
    plt.ylabel('Average Consumption Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    avg_profile_filename = os.path.join(output_dir, f'comparative_avg_consumption_profiles_{reg_id}_vs_{non_reg_id}.png')
    plt.savefig(avg_profile_filename)
    plt.close()
    print(f"Comparative average consumption profiles plot saved to '{avg_profile_filename}'")

    print("\nComparative analysis complete.")


if __name__ == '__main__':
    # --- Configuration for Comparative Analysis ---
    # IMPORTANT: Adjust these paths and parameters for your setup

    # Paths for the REGULAR (non-fraudulent) supply
    REGULAR_SUPPLY_ID = 'SUPPLY021' # Example: Choose a supply you know is regular
    REGULAR_SUPPLY_ANOMALY_FILE = f'/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Setup_Anomaly/Setup_Anomaly/src/Anomaly_Transformer/checkpoints/anomaly_scores/{REGULAR_SUPPLY_ID}_anomaly_scores.csv'
    REGULAR_SUPPLY_ORIGINAL_DATA_FILE = f'/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Study_datasets/Consumption/pivoted_data/{REGULAR_SUPPLY_ID}.csv'

    # Paths for the NON-REGULAR (potentially fraudulent/anomalous) supply
    NON_REGULAR_SUPPLY_ID = 'SUPPLY009' # Example: The one you just tested
    NON_REGULAR_SUPPLY_ANOMALY_FILE = f'/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Setup_Anomaly/Setup_Anomaly/src/Anomaly_Transformer/checkpoints/anomaly_scores/SUPPLY009_anomaly_scores.csv'
    NON_REGULAR_SUPPLY_ORIGINAL_DATA_FILE = f'/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Study_datasets/Consumption/pivoted_data/{NON_REGULAR_SUPPLY_ID}.csv'

    # The anomaly ratio used in solver.py when generating these scores for BOTH supplies
    ANOMALY_RATIO_USED = 4.0 

    # The test_size and random_state used in your EnergySegLoader's train_test_split
    # These MUST match the values used when generating the anomaly scores!
    TEST_SPLIT_SIZE = 0.2
    RANDOM_SPLIT_STATE = 42

    # Run the comparative analysis
    compare_supply_insights(
        regular_supply_anomaly_file=REGULAR_SUPPLY_ANOMALY_FILE,
        regular_supply_original_data_file=REGULAR_SUPPLY_ORIGINAL_DATA_FILE,
        non_regular_supply_anomaly_file=NON_REGULAR_SUPPLY_ANOMALY_FILE,
        non_regular_supply_original_data_file=NON_REGULAR_SUPPLY_ORIGINAL_DATA_FILE,
        anormly_ratio=ANOMALY_RATIO_USED,
        test_size_split=TEST_SPLIT_SIZE,
        random_state_split=RANDOM_SPLIT_STATE
    )
