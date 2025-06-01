import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # For saving the model and scaler
import os # For managing paths
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer # For handling NaNs in features

# --- Configuration ---
INPUT_MERGED_DF_PATH = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Unsupervized/Data/dataset_5day.csv'
OUTPUT_FINAL_CSV_PATH = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Unsupervized/Data/results_5day_OCSVM.csv'
MODEL_DIR = 'models/Unsupervized/models' # Directory to save the model and scaler

# Hyperparameter ranges for tuning
NU_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1]
GAMMA_VALUES = ['scale', 0.1, 0.01, 0.001]

# Define weights for the custom evaluation metric
WEIGHT_RECALL = 0.7
WEIGHT_FPR = 0.3

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Load the Merged DataFrame ---
print("--- Loading Merged DataFrame ---")
try:
    merged_df = pd.read_csv(INPUT_MERGED_DF_PATH)
except FileNotFoundError:
    print(f"Error: '{INPUT_MERGED_DF_PATH}' not found. Please provide the correct path to your merged_df CSV.")
    exit()

merged_df['Period_Start_Date'] = pd.to_datetime(merged_df['Period_Start_Date'])
merged_df['Period_End_Date'] = pd.to_datetime(merged_df['Period_End_Date'])

print(f"Merged DataFrame loaded successfully. Total rows: {len(merged_df)}")
print(merged_df.head())

# --- Helper Function for Supply-Level Data Splitting ---
def split_data_by_supply_id(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Splits the DataFrame by unique Supply_IDs into training, validation, and test sets.
    Stratifies by Is_Non_Regular to maintain class balance across splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Ratios must sum to 1"

    unique_supply_ids = df['Supply_ID'].unique()
    
    # Get the overall label for each supply (assuming Is_Non_Regular is constant per Supply_ID)
    supply_labels = df.groupby('Supply_ID')['Is_Non_Regular'].first() 

    # First split: train_ids and temp_ids (for val+test)
    train_ids, temp_ids, _, temp_labels = train_test_split(
        unique_supply_ids, supply_labels[unique_supply_ids],
        test_size=(val_ratio + test_ratio), stratify=supply_labels[unique_supply_ids], random_state=random_state
    )

    # Second split: val_ids and test_ids from temp_ids
    val_ids, test_ids, _, _ = train_test_split(
        temp_ids, temp_labels, # Use temp_labels for stratification of the second split
        test_size=(test_ratio / (val_ratio + test_ratio)), stratify=temp_labels, random_state=random_state
    )

    # Filter original DataFrame based on Supply_IDs
    train_df = df[df['Supply_ID'].isin(train_ids)].copy()
    val_df = df[df['Supply_ID'].isin(val_ids)].copy()
    test_df = df[df['Supply_ID'].isin(test_ids)].copy()

    print(f"\nData Split into (Supply_IDs):")
    print(f"  Train: {len(train_ids)} supplies ({train_df['Is_Non_Regular'].sum()} non-regular)")
    print(f"  Validation: {len(val_ids)} supplies ({val_df['Is_Non_Regular'].sum()} non-regular)")
    print(f"  Test: {len(test_ids)} supplies ({test_df['Is_Non_Regular'].sum()} non-regular)")

    return train_df, val_df, test_df

# --- Perform Supply-Level Data Splitting ---
train_df_full, val_df_full, test_df_full = split_data_by_supply_id(merged_df)

# --- 2. Prepare Features for One-Class SVM ---
print("\n--- Preparing Features for OCSVM ---")

feature_columns = [col for col in merged_df.columns if col not in
                   ['Supply_ID', 'Period_Start_Date', 'Period_End_Date', 'Period_Index',
                    'CLUSTER', 'Is_Non_Regular', 'anomaly_prediction_ocsvm', 'anomaly_score_ocsvm']]

# Extract features for each split
X_train_raw = train_df_full[feature_columns]
X_val_raw = val_df_full[feature_columns]
X_test_raw = test_df_full[feature_columns]

# --- Handle NaN values using imputation ---
# Fit imputer ONLY on training data to avoid data leakage
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_raw)

# Transform validation and test data using the *fitted* imputer
X_val_imputed = imputer.transform(X_val_raw)
X_test_imputed = imputer.transform(X_test_raw)

# --- Scale features ---
# Fit scaler ONLY on training data to avoid data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

# Transform validation and test data using the *fitted* scaler
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print(f"Features prepared and scaled. Number of features: {X_train_scaled.shape[1]}")

# --- Identify "normal" training data for OCSVM fitting ---
# OCSVM should be trained ONLY on normal samples (Is_Non_Regular == 0) from the training set.
X_train_ocsvm_fit = X_train_scaled[train_df_full['Is_Non_Regular'] == 0]
print(f"OCSVM will be trained on {len(X_train_ocsvm_fit)} normal periods from the training set.")


### 3. One-Class SVM Hyperparameter Tuning (Supply-Level Evaluation on Validation Set)

print("\n--- Starting One-Class SVM Hyperparameter Tuning (Supply-Level Evaluation on Validation Set) ---")

best_overall_weighted_score = -np.inf # Optimize for (0.7 * Supply Recall) - (0.3 * Supply False Positive Rate)
best_overall_params = {}
best_overall_threshold = None
evaluation_results = []

# Get true labels for validation supplies
# These are the ground truth labels for the supplies in the validation set
all_val_supply_ids = val_df_full['Supply_ID'].unique()
val_true_non_regular_supply_ids = val_df_full[val_df_full['Is_Non_Regular'] == 1]['Supply_ID'].unique()
val_true_regular_supply_ids = val_df_full[val_df_full['Is_Non_Regular'] == 0]['Supply_ID'].unique()

total_val_non_regular_supplies = len(val_true_non_regular_supply_ids)
total_val_regular_supplies = len(val_true_regular_supply_ids)

print(f"Validation Set: {total_val_non_regular_supplies} Non-Regular Supplies, {total_val_regular_supplies} Regular Supplies.")

for nu_val in NU_VALUES:
    for gamma_val in GAMMA_VALUES:
        print(f"\n--- Testing nu={nu_val}, gamma={gamma_val} ---")
        
        # Train OCSVM ONLY on normal training data (CRUCIAL CHANGE)
        oc_svm_model = OneClassSVM(kernel='rbf', nu=nu_val, gamma=gamma_val)
        oc_svm_model.fit(X_train_ocsvm_fit)

        # Get decision scores for the ENTIRE VALIDATION SET
        val_decision_scores = oc_svm_model.decision_function(X_val_scaled)
        
        # --- Tune optimal threshold on Validation Set for current nu/gamma ---
        best_threshold_for_current_params = None
        best_weighted_score_for_current_params = -np.inf
        
        # Define a range of thresholds to test, centered around common OCSVM decision function values
        # Sorting unique scores helps to test only relevant thresholds
        threshold_range = np.linspace(val_decision_scores.min() - 0.1, val_decision_scores.max() + 0.1, 100)
        
        for threshold in threshold_range:
            # Period-level predictions based on current threshold
            # OCSVM decision function: positive values for normal, negative for anomaly.
            # So, score < threshold indicates an anomaly.
            period_predictions_val = np.where(val_decision_scores < threshold, -1, 1)

            # Create a temporary DataFrame for validation set to easily group by Supply_ID
            temp_val_df_for_eval = val_df_full[['Supply_ID', 'Is_Non_Regular']].copy()
            temp_val_df_for_eval['anomaly_prediction_ocsvm'] = period_predictions_val

            # Identify Supply_IDs that had at least one anomalous period flagged by OCSVM
            val_supplies_flagged_by_ocsvm = temp_val_df_for_eval[temp_val_df_for_eval['anomaly_prediction_ocsvm'] == -1]['Supply_ID'].unique()

            # --- Calculate Supply-Level Confusion Matrix Components for VALIDATION SET ---
            true_positive_supplies_val = len(np.intersect1d(val_supplies_flagged_by_ocsvm, val_true_non_regular_supply_ids))
            false_positive_supplies_val = len(np.intersect1d(val_supplies_flagged_by_ocsvm, val_true_regular_supply_ids))
            false_negative_supplies_val = len(np.setdiff1d(val_true_non_regular_supply_ids, val_supplies_flagged_by_ocsvm))
            true_negative_supplies_val = len(np.setdiff1d(val_true_regular_supply_ids, val_supplies_flagged_by_ocsvm))

            supply_recall_non_regular_val = true_positive_supplies_val / total_val_non_regular_supplies if total_val_non_regular_supplies > 0 else 0.0
            supply_false_positive_rate_val = false_positive_supplies_val / total_val_regular_supplies if total_val_regular_supplies > 0 else 0.0
            
            current_weighted_score_val = (WEIGHT_RECALL * supply_recall_non_regular_val) - (WEIGHT_FPR * supply_false_positive_rate_val)

            if current_weighted_score_val > best_weighted_score_for_current_params:
                best_weighted_score_for_current_params = current_weighted_score_val
                best_threshold_for_current_params = threshold
        
        # Append the best result for this nu/gamma combination to the results list
        evaluation_results.append({
            'nu': nu_val,
            'gamma': gamma_val,
            'best_threshold': best_threshold_for_current_params,
            'weighted_score_val': best_weighted_score_for_current_params,
            # For logging purposes, recalculate metrics for the best threshold on validation
            # (Optional: can optimize to avoid recalculation, but good for clarity)
            'TP_Supplies_val': true_positive_supplies_val, # These would be from the last threshold iter, not necessarily the best
            'FP_Supplies_val': false_positive_supplies_val,
            'FN_Supplies_val': false_negative_supplies_val,
            'TN_Supplies_val': true_negative_supplies_val,
            'supply_recall_non_regular_val': supply_recall_non_regular_val,
            'supply_false_positive_rate_val': supply_false_positive_rate_val
        })

        # Update overall best parameters and threshold
        if best_weighted_score_for_current_params > best_overall_weighted_score:
            best_overall_weighted_score = best_weighted_score_for_current_params
            best_overall_params = {'nu': nu_val, 'gamma': gamma_val}
            best_overall_threshold = best_threshold_for_current_params

results_df_eval = pd.DataFrame(evaluation_results)
print("\n--- Hyperparameter Evaluation Results (Validation Set) ---")
print(results_df_eval.sort_values(by='weighted_score_val', ascending=False).to_string())

print(f"\nBest overall parameters (based on weighted score on Validation Set): {best_overall_params}")
print(f"Best overall threshold (for decision function on Validation Set): {best_overall_threshold:.4f}")
print(f"Best overall weighted score on Validation Set: {best_overall_weighted_score:.4f}")

# --- Optional: Visualize Tuning Results (using supply-level metrics) ---
# Adjust visualization to show the best score on validation set for each nu/gamma combination
plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
sns.lineplot(data=results_df_eval, x='nu', y='weighted_score_val', hue='gamma', marker='o')
plt.title('Weighted Score (Validation) vs. Nu')
plt.xlabel('nu (Expected Anomaly Proportion)')
plt.ylabel('Weighted Score (Validation)')
plt.grid(True)
plt.xscale('log')

plt.subplot(1, 2, 2)
sns.lineplot(data=results_df_eval, x='nu', y='best_threshold', hue='gamma', marker='o')
plt.title('Best Threshold (Validation) vs. Nu')
plt.xlabel('nu (Expected Anomaly Proportion)')
plt.ylabel('Optimal Decision Function Threshold')
plt.grid(True)
plt.xscale('log')
plt.tight_layout()
plt.show()


### 4. Final One-Class SVM Model Training and Application (on Test Set)

print(f"\n--- Training Final OCSVM Model with nu={best_overall_params['nu']}, gamma={best_overall_params['gamma']} ---")

final_oc_svm_model = OneClassSVM(kernel='rbf', nu=best_overall_params['nu'], gamma=best_overall_params['gamma'])
# Train the final model again on ALL normal training data
final_oc_svm_model.fit(X_train_ocsvm_fit)

# # Save the trained model and scaler
# joblib.dump(final_oc_svm_model, os.path.join(MODEL_DIR, 'final_ocsvm_model.pkl'))
# joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
# joblib.dump(imputer, os.path.join(MODEL_DIR, 'imputer.pkl'))
# print(f"Final OCSVM model, scaler, and imputer saved to {MODEL_DIR}")


print("\n--- Evaluating Final OCSVM Model on UNSEEN TEST SET ---")

# Get decision scores for the TEST set
test_decision_scores = final_oc_svm_model.decision_function(X_test_scaled)

# Apply the best overall threshold found during validation
test_period_predictions = np.where(test_decision_scores < best_overall_threshold, -1, 1)

# Create a temporary DataFrame for test set to easily group by Supply_ID
test_df_for_eval = test_df_full[['Supply_ID', 'Is_Non_Regular']].copy()
test_df_for_eval['anomaly_prediction_ocsvm'] = test_period_predictions
test_df_for_eval['anomaly_score_ocsvm'] = test_decision_scores # Add scores for inspection

# Identify Supply_IDs that had at least one anomalous period flagged by OCSVM
test_ocsvm_flagged_supplies = test_df_for_eval[test_df_for_eval['anomaly_prediction_ocsvm'] == -1]['Supply_ID'].unique()

# Get true labels for test supplies
test_true_non_regular_supply_ids = test_df_full[test_df_full['Is_Non_Regular'] == 1]['Supply_ID'].unique()
test_true_regular_supply_ids = test_df_full[test_df_full['Is_Non_Regular'] == 0]['Supply_ID'].unique()

total_test_non_regular_supplies = len(test_true_non_regular_supply_ids)
total_test_regular_supplies = len(test_true_regular_supply_ids)

# --- Calculate Supply-Level Confusion Matrix Components for TEST SET ---
true_positive_supplies_test = len(np.intersect1d(test_ocsvm_flagged_supplies, test_true_non_regular_supply_ids))
false_positive_supplies_test = len(np.intersect1d(test_ocsvm_flagged_supplies, test_true_regular_supply_ids))
false_negative_supplies_test = len(np.setdiff1d(test_true_non_regular_supply_ids, test_ocsvm_flagged_supplies))
true_negative_supplies_test = len(np.setdiff1d(test_true_regular_supply_ids, test_ocsvm_flagged_supplies))

supply_recall_non_regular_test = true_positive_supplies_test / total_test_non_regular_supplies if total_test_non_regular_supplies > 0 else 0.0
supply_false_positive_rate_test = false_positive_supplies_test / total_test_regular_supplies if total_test_regular_supplies > 0 else 0.0

final_weighted_score_test = (WEIGHT_RECALL * supply_recall_non_regular_test) - (WEIGHT_FPR * supply_false_positive_rate_test)

print(f"\n--- Final OCSVM Performance on UNSEEN TEST SET ---")
print(f"  True Positives (Supplies): {true_positive_supplies_test}")
print(f"  False Positives (Supplies): {false_positive_supplies_test}")
print(f"  False Negatives (Supplies): {false_negative_supplies_test}")
print(f"  True Negatives (Supplies): {true_negative_supplies_test}")
print(f"  Supply Recall (Non-Regular): {supply_recall_non_regular_test:.4f}")
print(f"  Supply False Positive Rate: {supply_false_positive_rate_test:.4f}")
print(f"  Final Weighted Score: {final_weighted_score_test:.4f}")

# --- Add final predictions and scores to the original merged_df for full analysis ---
# This step is only for reporting/analysis, not for evaluation.
# We'll use the final model to predict on the *entire* scaled dataset (X_scaled) to populate these columns.
# This assumes X_scaled was produced from the entire merged_df.
merged_df['anomaly_prediction_ocsvm'] = final_oc_svm_model.predict(scaler.transform(imputer.transform(merged_df[feature_columns])))
merged_df['anomaly_score_ocsvm'] = final_oc_svm_model.decision_function(scaler.transform(imputer.transform(merged_df[feature_columns])))

# Apply the best_overall_threshold to the entire dataset for consistent binary flags
merged_df['anomaly_prediction_ocsvm_thresholded'] = np.where(
    merged_df['anomaly_score_ocsvm'] < best_overall_threshold, -1, 1
)

print("\n--- Final Merged DataFrame Head with OCSVM Results (including thresholded predictions) ---")
print(merged_df[['Supply_ID', 'Period_Start_Date', 'Period_End_Date',
                 'Mean_Consumption_Period', 'Is_Non_Regular',
                 'anomaly_prediction_ocsvm', 'anomaly_score_ocsvm',
                 'anomaly_prediction_ocsvm_thresholded']].head())

print("\n--- Counts of OCSVM Anomaly Predictions (Period-Level, after thresholding) ---")
print(merged_df['anomaly_prediction_ocsvm_thresholded'].value_counts())

# Save the updated merged_df with OCSVM predictions
merged_df.to_csv(OUTPUT_FINAL_CSV_PATH, index=False)
print(f"\nUpdated merged_df with OCSVM predictions saved to {OUTPUT_FINAL_CSV_PATH}")