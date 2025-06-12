import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Needed to re-create scaler for original data
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def analyze_single_supply_scores(
    anomaly_score_filepath: str,
    original_supply_data_filepath: str, # New parameter for original consumption data
    anormly_ratio: float,
    output_dir: str = "single_supply_analysis_results"
):
    """
    Loads and analyzes anomaly scores for a single supply, combining with original consumption data.

    Args:
        anomaly_score_filepath (str): Full path to the anomaly score CSV file for one supply.
        original_supply_data_filepath (str): Full path to the original, unscaled consumption CSV
                                             for the same supply (e.g., SUPPLY009.csv).
        anormly_ratio (float): The anomaly ratio (e.g., 4.0 for 4%) used to determine
                               the threshold during the Anomaly Transformer's testing phase.
        output_dir (str): Directory to save analysis results (plots).
    """

    os.makedirs(output_dir, exist_ok=True)
    supply_id = os.path.basename(anomaly_score_filepath).replace("_anomaly_scores.csv", "")
    print(f"Analyzing anomaly scores for supply: {supply_id}")
    print(f"Analysis results will be saved to: {os.path.abspath(output_dir)}")

    # --- 1. Load Anomaly Scores ---
    print("\n--- Loading anomaly scores ---")
    try:
        # Anomaly scores are expected to be a single column of floats
        scores = pd.read_csv(anomaly_score_filepath, header=None).iloc[:, 0].values
        if len(scores) == 0:
            print(f"Error: Anomaly score file '{anomaly_score_filepath}' is empty. Exiting.")
            return
        print(f"Loaded {len(scores)} anomaly scores.")
    except FileNotFoundError:
        print(f"Error: Anomaly score file not found at '{anomaly_score_filepath}'. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading anomaly scores from '{anomaly_score_filepath}': {e}")
        return

    # --- 2. Load and Prepare Original Consumption Data ---
    print("\n--- Loading original consumption data ---")
    try:
        # Load the original CSV, drop 'date', and handle comma decimals
        full_original_data = pd.read_csv(original_supply_data_filepath).drop(columns=["date"]).apply(lambda col: col.str.replace(',', '.').astype(float))
        
        # Re-create the train/test split exactly as in EnergySegLoader
        # This ensures anomaly scores align with the correct days in data_test
        data_train_original, data_test_original = train_test_split(
            full_original_data, test_size=0.2, random_state=42 # Must match EnergySegLoader's split
        )
        # Ensure the test data length matches the anomaly scores length
        if len(data_test_original) != len(scores):
            print(f"Warning: Length mismatch between original test data ({len(data_test_original)} days) "
                  f"and anomaly scores ({len(scores)} scores). This might affect alignment.")
            # Trim or pad data_test_original if necessary for plotting, or adjust logic
            # For now, we'll proceed, but be aware of potential misalignment if lengths differ
            data_test_original = data_test_original.iloc[:len(scores)]

        print(f"Loaded original consumption data for {len(full_original_data)} days.")
        print(f"Extracted test set consumption data for {len(data_test_original)} days.")

    except FileNotFoundError:
        print(f"Error: Original consumption data file not found at '{original_supply_data_filepath}'. Please check the path.")
        return
    except Exception as e:
        print(f"Error loading original consumption data from '{original_supply_data_filepath}': {e}")
        return

    # --- 3. Calculate Threshold ---
    threshold = np.percentile(scores, 100 - anormly_ratio)
    print(f"Calculated anomaly threshold (using {anormly_ratio}%): {threshold:.6f}")

    # --- 4. Descriptive Statistics ---
    print("\n--- Descriptive Statistics of Anomaly Scores ---")
    scores_series = pd.Series(scores)
    print(scores_series.describe())
    print(f"Number of zeros: {np.sum(scores == 0)}")
    print(f"Number of non-zeros: {np.sum(scores != 0)}")

    # --- 5. Identify Anomalous Days ---
    anomalous_days_binary = (scores > threshold).astype(int)
    num_anomalous_days = np.sum(anomalous_days_binary)
    percentage_anomalous = (num_anomalous_days / len(scores)) * 100
    print(f"\nNumber of anomalous days identified (scores > threshold): {num_anomalous_days}")
    print(f"Percentage of anomalous days: {percentage_anomalous:.2f}%")

    # --- 6. Visualization: Anomaly Scores Over Time ---
    print("\n--- Generating Anomaly Score Plot ---")
    plt.figure(figsize=(15, 7))
    time_index = np.arange(len(scores)) # Represents days in the test set

    plt.plot(time_index, scores, label='Anomaly Score', color='blue', alpha=0.7)
    anomalous_indices = np.where(anomalous_days_binary == 1)[0]
    plt.scatter(anomalous_indices, scores[anomalous_indices], color='red', s=50, zorder=5, label='Anomalous Day')
    plt.axhline(y=threshold, color='green', linestyle='--', label=f'Threshold ({threshold:.2f})')

    plt.title(f'Anomaly Scores Over Time for Supply: {supply_id}')
    plt.xlabel('Day Index in Test Set')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'{supply_id}_anomaly_scores_plot.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Anomaly score plot saved to '{plot_filename}'")

    # --- 7. Visualization: Histogram of Anomaly Scores ---
    print("\n--- Generating Anomaly Score Histogram ---")
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=50, kde=True, color='purple')
    plt.axvline(x=threshold, color='green', linestyle='--', label=f'Threshold ({threshold:.2f})')
    plt.title(f'Distribution of Anomaly Scores for Supply: {supply_id}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    hist_filename = os.path.join(output_dir, f'{supply_id}_anomaly_scores_histogram.png')
    plt.savefig(hist_filename)
    plt.close()
    print(f"Anomaly score histogram saved to '{hist_filename}'")

    # --- 8. Combined Visualization: Anomalous Days Consumption Patterns ---
    print("\n--- Generating Combined Consumption & Anomaly Plots ---")

    # Select a few top anomalous days for detailed plotting
    # Sort scores in descending order and get original indices
    sorted_indices = np.argsort(scores)[::-1]
    top_anomalous_day_indices = [idx for idx in sorted_indices if scores[idx] > threshold][:5] # Top 5 anomalous days

    if len(top_anomalous_day_indices) > 0:
        print(f"Plotting consumption profiles for top {len(top_anomalous_day_indices)} anomalous days...")
        for i, day_idx in enumerate(top_anomalous_day_indices):
            plt.figure(figsize=(12, 6))
            # Get the consumption profile for this specific anomalous day
            consumption_profile = data_test_original.iloc[day_idx].values
            
            plt.plot(np.arange(1, 97), consumption_profile, color='blue', label='Consumption (Quarter-Hours)')
            plt.title(f'Consumption Profile for Anomalous Day Index {day_idx} (Anomaly Score: {scores[day_idx]:.4f})')
            plt.xlabel('Quarter-Hour of Day (1-96)')
            plt.ylabel('Consumption Value')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            combined_plot_filename = os.path.join(output_dir, f'{supply_id}_anomalous_day_{day_idx}_consumption.png')
            plt.savefig(combined_plot_filename)
            plt.close()
            print(f"  Saved consumption plot for anomalous day {day_idx} to '{combined_plot_filename}'")
    else:
        print("No anomalous days found above threshold to plot consumption profiles.")

    # --- 9. Average Consumption Profiles: Anomalous vs. Normal ---
    print("\n--- Generating Average Consumption Profiles ---")
    
    # Get consumption data for normal and anomalous days
    normal_day_indices = np.where(anomalous_days_binary == 0)[0]
    
    if len(normal_day_indices) > 0:
        avg_normal_profile = data_test_original.iloc[normal_day_indices].mean(axis=0)
    else:
        avg_normal_profile = np.zeros(data_test_original.shape[1]) # All zeros if no normal days

    if num_anomalous_days > 0:
        avg_anomalous_profile = data_test_original.iloc[anomalous_indices].mean(axis=0)
    else:
        avg_anomalous_profile = np.zeros(data_test_original.shape[1]) # All zeros if no anomalous days

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, 97), avg_normal_profile, label='Average Normal Day Profile', color='green', linestyle='-')
    if num_anomalous_days > 0:
        plt.plot(np.arange(1, 97), avg_anomalous_profile, label='Average Anomalous Day Profile', color='red', linestyle='--')
    
    plt.title(f'Average Daily Consumption Profile: Normal vs. Anomalous for Supply: {supply_id}')
    plt.xlabel('Quarter-Hour of Day (1-96)')
    plt.ylabel('Average Consumption Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    avg_profile_filename = os.path.join(output_dir, f'{supply_id}_avg_consumption_profiles.png')
    plt.savefig(avg_profile_filename)
    plt.close()
    print(f"Average consumption profiles plot saved to '{avg_profile_filename}'")

    print("\nAnalysis for single supply complete.")

if __name__ == '__main__':
    # --- Configuration for a Single Supply ---
    # IMPORTANT: Adjust these paths and parameters for your setup

    # Path to the specific anomaly score CSV file for SUPPLY009
    # Example: 'Setup_Anomaly/src/Anomaly_Transformer/checkpoints/anomaly_scores/SUPPLY009_anomaly_scores.csv'
    SUPPLY_009_ANOMALY_FILE = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Setup_Anomaly/Setup_Anomaly/src/Anomaly_Transformer/checkpoints/anomaly_scores/SUPPLY021_anomaly_scores.csv'

    # Path to the original, full consumption data CSV for SUPPLY009
    # This should be the file that was read by your EnergySegLoader before splitting.
    # Example: '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Study_datasets/Consumption/pivoted_data/SUPPLY009.csv'
    ORIGINAL_SUPPLY_009_DATA_FILE = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Study_datasets/Consumption/pivoted_data/SUPPLY021.csv'

    # The anomaly ratio used in solver.py when generating these scores
    ANOMALY_RATIO_FOR_THIS_SUPPLY = 4.0 

    # Run the analysis for SUPPLY009
    analyze_single_supply_scores(
        anomaly_score_filepath=SUPPLY_009_ANOMALY_FILE,
        original_supply_data_filepath=ORIGINAL_SUPPLY_009_DATA_FILE,
        anormly_ratio=ANOMALY_RATIO_FOR_THIS_SUPPLY
    )