import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def analyze_anomaly_scores(
    anomaly_scores_dir: str,
    supply_fraud_labels_path: str,
    anormly_ratio: float = 4.0, # The anomaly ratio used during model testing
    output_dir: str = "analysis_results"
):
    """
    Loads anomaly scores, engineers features, combines with fraud labels,
    and performs analysis/classification.

    Args:
        anomaly_scores_dir (str): Directory containing saved anomaly score CSVs.
        supply_fraud_labels_path (str): Path to the CSV with supply-level fraud labels.
                                        Expected columns: 'supply_id', 'is_fraud' (0 or 1).
        anormly_ratio (float): The anomaly ratio (e.g., 4.0 for 4%) used to determine
                               the threshold during the Anomaly Transformer's testing phase.
        output_dir (str): Directory to save analysis results (plots, combined data).
    """

    os.makedirs(output_dir, exist_ok=True)
    print(f"Analysis results will be saved to: {os.path.abspath(output_dir)}")

    # --- 1. Load Supply-Level Fraud Labels ---
    print("\n--- Loading supply-level fraud labels ---")
    try:
        supply_labels_df = pd.read_csv(supply_fraud_labels_path)
        if 'supply_id' not in supply_labels_df.columns or 'is_fraud' not in supply_labels_df.columns:
            raise ValueError("Supply fraud labels CSV must contain 'supply_id' and 'is_fraud' columns.")
        supply_labels_df['supply_id'] = supply_labels_df['supply_id'].astype(str) # Ensure string type
        print(f"Loaded {len(supply_labels_df)} supply labels.")
        print(supply_labels_df['is_fraud'].value_counts())
    except Exception as e:
        print(f"Error loading supply fraud labels: {e}")
        print("Please ensure the path is correct and the CSV has 'supply_id' and 'is_fraud' columns.")
        return

    # --- 2. Load Anomaly Scores and Engineer Features ---
    print("\n--- Processing anomaly scores and engineering features ---")
    all_supply_features = []

    for filename in os.listdir(anomaly_scores_dir):
        if filename.endswith("_anomaly_scores.csv"):
            supply_id = filename.replace("_anomaly_scores.csv", "")
            filepath = os.path.join(anomaly_scores_dir, filename)

            try:
                # Anomaly scores are expected to be a single column of floats
                scores = pd.read_csv(filepath, header=None).iloc[:, 0].values
                
                if len(scores) == 0:
                    print(f"Skipping {supply_id}: Anomaly score file is empty.")
                    continue

                # Calculate threshold for this supply's scores (consistent with Solver)
                # Note: The Solver used combined train/test energy for threshold.
                # For simplicity here, we'll use the test scores directly for thresholding,
                # or you can pass the global threshold if you saved it.
                # Using percentile on the current supply's scores for feature engineering
                # is a reasonable proxy if the global threshold isn't easily accessible.
                current_thresh = np.percentile(scores, 100 - anormly_ratio)

                # Feature Engineering
                max_anomaly_score = np.max(scores)
                mean_anomaly_score = np.mean(scores)
                std_anomaly_score = np.std(scores)
                
                # Identify anomalous points based on the threshold
                anomalous_points = scores[scores > current_thresh]
                num_anomalous_days = len(anomalous_points)
                
                # Handle case where no points are above threshold
                mean_score_above_threshold = np.mean(anomalous_points) if num_anomalous_days > 0 else 0
                sum_score_above_threshold = np.sum(anomalous_points) if num_anomalous_days > 0 else 0
                
                percentage_anomalous_days = (num_anomalous_days / len(scores)) * 100 if len(scores) > 0 else 0

                # Longest consecutive anomaly duration
                # This requires iterating through the binary predictions
                binary_preds = (scores > current_thresh).astype(int)
                longest_anomaly_duration = 0
                current_duration = 0
                for p in binary_preds:
                    if p == 1:
                        current_duration += 1
                    else:
                        longest_anomaly_duration = max(longest_anomaly_duration, current_duration)
                        current_duration = 0
                longest_anomaly_duration = max(longest_anomaly_duration, current_duration) # Catch trailing anomaly

                all_supply_features.append({
                    'supply_id': supply_id,
                    'max_anomaly_score': max_anomaly_score,
                    'mean_anomaly_score': mean_anomaly_score,
                    'std_anomaly_score': std_anomaly_score,
                    'num_anomalous_days': num_anomalous_days,
                    'percentage_anomalous_days': percentage_anomalous_days,
                    'mean_score_above_threshold': mean_score_above_threshold,
                    'sum_score_above_threshold': sum_score_above_threshold,
                    'longest_anomaly_duration': longest_anomaly_duration
                })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    if not all_supply_features:
        print("No anomaly score files found or processed. Exiting.")
        return

    features_df = pd.DataFrame(all_supply_features)
    print(f"Engineered features for {len(features_df)} supplies.")
    # print(features_df.head())

    # --- 3. Merge with Ground Truth Labels ---
    print("\n--- Merging features with fraud labels ---")
    merged_df = pd.merge(features_df, supply_labels_df, on='supply_id', how='left')

    # Handle supplies that might not have a fraud label (e.g., if some files were processed but not in label CSV)
    initial_len = len(merged_df)
    merged_df.dropna(subset=['is_fraud'], inplace=True)
    merged_df['is_fraud'] = merged_df['is_fraud'].astype(int)
    if len(merged_df) < initial_len:
        print(f"Warning: Dropped {initial_len - len(merged_df)} supplies due to missing fraud labels.")
    
    print(f"Combined data for {len(merged_df)} supplies.")
    # print(merged_df.head())

    # --- 4. Analysis and Insights ---
    print("\n--- Performing analysis and generating insights ---")

    # Compare features for fraudulent vs. non-fraudulent supplies
    print("\nDescriptive Statistics (Fraudulent vs. Non-Fraudulent):")
    for col in features_df.columns:
        if col == 'supply_id':
            continue
        print(f"\n--- {col} ---")
        print(merged_df.groupby('is_fraud')[col].describe())

        # Visualization: Box plot
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='is_fraud', y=col, data=merged_df)
        plt.title(f'Distribution of {col} by Fraud Status')
        plt.xlabel('Is Fraud (0: No, 1: Yes)')
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'boxplot_{col}.png'))
        plt.close()

    print(f"\nBox plots saved to '{output_dir}' for visual comparison.")

    # --- 5. Train a Simple Classifier (Optional) ---
    print("\n--- Training a simple classifier for fraud detection ---")
    
    X = merged_df.drop(columns=['supply_id', 'is_fraud'])
    y = merged_df['is_fraud']

    if len(np.unique(y)) < 2:
        print("Cannot train classifier: Only one class present in fraud labels.")
        return

    # Ensure enough samples for train/test split
    if len(X) < 2:
        print("Not enough samples to train a classifier.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    print(f"Fraud distribution in training: {y_train.value_counts(normalize=True)}")
    print(f"Fraud distribution in testing: {y_test.value_counts(normalize=True)}")

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n--- Classifier Performance (on engineered features) ---")
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("  [[TN, FP]")
    print("   [FN, TP]]")

    # Feature Importance
    print("\n--- Feature Importance from Classifier ---")
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(feature_importances)

    # Save the combined feature dataset
    merged_df.to_csv(os.path.join(output_dir, "combined_anomaly_features_with_fraud_labels.csv"), index=False)
    print(f"\nCombined feature dataset saved to '{output_dir}/combined_anomaly_features_with_fraud_labels.csv'")
    print("Analysis complete.")

if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Adjust these paths and parameters for your setup
    
    # Directory where anomaly scores are saved by solver.py
    # Example: 'Setup_Anomaly/src/Anomaly_Transformer/checkpoints/anomaly_scores'
    ANOMALY_SCORES_DIRECTORY = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Setup_Anomaly/src/Anomaly_Transformer/checkpoints/anomaly_scores' 
    
    # Path to your CSV file containing supply-level fraud labels
    # This file should have columns like 'supply_id' and 'is_fraud' (0 or 1)
    # Example: 'path/to/your/supply_fraud_labels.csv'
    # For demonstration, I'm creating a dummy one. REPLACE THIS WITH YOUR REAL FILE!
    DUMMY_FRAUD_LABELS_PATH = 'dummy_supply_fraud_labels.csv'
    
    # Create a dummy fraud labels file for testing if it doesn't exist
    if not os.path.exists(DUMMY_FRAUD_LABELS_PATH):
        print(f"Creating a dummy fraud labels file at {DUMMY_FRAUD_LABELS_PATH} for demonstration.")
        # This assumes you have supply IDs like SUPPLY001, SUPPLY002, etc.
        # You'll need to replace this with your actual fraud labels.
        dummy_data = {
            'supply_id': [f'SUPPLY{i:03d}' for i in range(1, 101)], # Example for 100 supplies
            'is_fraud': np.random.randint(0, 2, 100) # Randomly assign 0 or 1 for demonstration
        }
        # Manually set a few to be fraudulent for better demo
        dummy_data['is_fraud'][0] = 1 # SUPPLY001 is fraudulent
        dummy_data['is_fraud'][10] = 1 # SUPPLY011 is fraudulent
        dummy_data['is_fraud'][20] = 1 # SUPPLY021 is fraudulent
        dummy_data['is_fraud'][30] = 1 # SUPPLY031 is fraudulent
        dummy_data['is_fraud'][33] = 1 # SUPPLY034 is fraudulent (as you tested this one)

        pd.DataFrame(dummy_data).to_csv(DUMMY_FRAUD_LABELS_PATH, index=False)
        print("Dummy file created. PLEASE REPLACE WITH YOUR ACTUAL FRAUD LABELS FILE!")


    # The anomaly ratio used in solver.py during testing
    ANOMALY_RATIO_USED_IN_SOLVER = 4.0 

    # Run the analysis
    analyze_anomaly_scores(
        anomaly_scores_dir=ANOMALY_SCORES_DIRECTORY,
        supply_fraud_labels_path=DUMMY_FRAUD_LABELS_PATH, # Use your real path here
        anormly_ratio=ANOMALY_RATIO_USED_IN_SOLVER
    )