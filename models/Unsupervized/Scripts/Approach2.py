import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # For saving the model and scaler
import os # For managing paths

# --- Configuration ---
# IMPORTANT: Update this path to where your merged_df CSV is saved
INPUT_MERGED_DF_PATH = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Unsupervized/Data/Approach1_OCSVM.csv' # Assuming you saved it with the previous script
OUTPUT_FINAL_CSV_PATH = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Unsupervized/Data/Approach2_OCSVM.csv' # A new name for the output
MODEL_DIR = 'models/Unsupervized/models' # Directory to save the model and scaler

# Hyperparameter ranges for tuning
NU_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1]
GAMMA_VALUES = ['scale', 0.1, 0.01, 0.001]

# --- 1. Load the Merged DataFrame ---
print("--- Loading Merged DataFrame ---")
try:
    merged_df = pd.read_csv(INPUT_MERGED_DF_PATH)
except FileNotFoundError:
    print(f"Error: '{INPUT_MERGED_DF_PATH}' not found. Please provide the correct path to your merged_df CSV.")
    exit()

# Ensure date columns are datetime objects, as they might be strings after loading from CSV
merged_df['Period_Start_Date'] = pd.to_datetime(merged_df['Period_Start_Date'])
merged_df['Period_End_Date'] = pd.to_datetime(merged_df['Period_End_Date'])

print(f"Merged DataFrame loaded successfully. Total rows: {len(merged_df)}")
print(merged_df.head())

# --- 2. Prepare Features for One-Class SVM ---
print("\n--- Preparing Features for OCSVM ---")

# Define features for OCSVM (exclude identifiers and target labels)
# 'CLUSTER' was likely used in the labels merging step, and 'Is_Non_Regular' is our evaluation target.
# Ensure 'anomaly_prediction_ocsvm' and 'anomaly_score_ocsvm' are not in feature_columns
# if they were added by a previous run and you're re-running this script.
feature_columns = [col for col in merged_df.columns if col not in
                   ['Supply_ID', 'Period_Start_Date', 'Period_End_Date', 'Period_Index',
                    'CLUSTER', 'Is_Non_Regular', 'anomaly_prediction_ocsvm', 'anomaly_score_ocsvm']]

X = merged_df[feature_columns]

# Scale features - essential for SVMs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features prepared and scaled. Number of features: {X_scaled.shape[1]}")


### 3. One-Class SVM Hyperparameter Tuning (Supply-Level Evaluation)

'This section performs the hyperparameter tuning using **supply-level evaluation metrics** to find the `nu` and `gamma` parameters that best align with your goal of identifying problematic supplies.'


print("\n--- Starting One-Class SVM Hyperparameter Tuning (Supply-Level Evaluation) ---")

best_score = -np.inf # Optimize for (0.7 * Supply Recall) - (0.3 * Supply False Positive Rate)
best_params = {}
evaluation_results = []

# Get unique Supply_IDs categorized by their overall label (Is_Non_Regular)
all_non_regular_supply_ids = merged_df[merged_df['Is_Non_Regular'] == 1]['Supply_ID'].unique()
all_regular_supply_ids = merged_df[merged_df['Is_Non_Regular'] == 0]['Supply_ID'].unique()

total_non_regular_supplies = len(all_non_regular_supply_ids)
total_regular_supplies = len(all_regular_supply_ids)

print(f"Total unique Non-Regular Supplies in dataset: {total_non_regular_supplies}")
print(f"Total unique Regular Supplies in dataset: {total_regular_supplies}")

for nu_val in NU_VALUES:
    for gamma_val in GAMMA_VALUES:
        print(f"\n--- Testing nu={nu_val}, gamma={gamma_val} ---")
        oc_svm_model = OneClassSVM(kernel='rbf', nu=nu_val, gamma=gamma_val)
        oc_svm_model.fit(X_scaled)

        # Get period-level predictions (-1 for anomaly, 1 for normal)
        period_predictions = oc_svm_model.predict(X_scaled)
        
        # Create a temporary DataFrame to easily group by Supply_ID and check for anomalies
        temp_df_for_eval = merged_df[['Supply_ID', 'Is_Non_Regular']].copy()
        temp_df_for_eval['anomaly_prediction_ocsvm'] = period_predictions

        # Identify Supply_IDs that had at least one anomalous period flagged by OCSVM
        supplies_flagged_by_ocsvm = temp_df_for_eval[temp_df_for_eval['anomaly_prediction_ocsvm'] == -1]['Supply_ID'].unique()

        # --- Calculate Supply-Level Confusion Matrix Components (Raw Counts) ---
        # TP_Supplies: Non-Regular Supplies correctly identified
        true_positive_supplies_count = len(np.intersect1d(supplies_flagged_by_ocsvm, all_non_regular_supply_ids))

        # FP_Supplies: Regular Supplies incorrectly flagged (potential new anomalies)
        false_positive_supplies_count = len(np.intersect1d(supplies_flagged_by_ocsvm, all_regular_supply_ids))

        # FN_Supplies: Non-Regular Supplies missed by OCSVM
        false_negative_supplies_count = len(np.setdiff1d(all_non_regular_supply_ids, supplies_flagged_by_ocsvm))
        
        # TN_Supplies: Regular Supplies correctly not flagged
        true_negative_supplies_count = len(np.setdiff1d(all_regular_supply_ids, supplies_flagged_by_ocsvm))

        # --- Calculate Supply-Level Metrics (Percentages) ---
        supply_recall_non_regular = true_positive_supplies_count / total_non_regular_supplies if total_non_regular_supplies > 0 else 0.0
        supply_false_positive_rate = false_positive_supplies_count / total_regular_supplies if total_regular_supplies > 0 else 0.0
        
        # Overall period anomaly rate (still useful to track overall sensitivity set by nu)
        overall_period_anomaly_rate = (period_predictions == -1).sum() / len(period_predictions) * 100

        evaluation_results.append({
            'nu': nu_val,
            'gamma': gamma_val,
            'overall_period_anomaly_rate_%': overall_period_anomaly_rate,
            # Raw Counts
            'TP_Supplies': true_positive_supplies_count,
            'FP_Supplies': false_positive_supplies_count,
            'FN_Supplies': false_negative_supplies_count,
            'TN_Supplies': true_negative_supplies_count,
            # Percentage Metrics
            'supply_recall_non_regular': supply_recall_non_regular,
            'supply_false_positive_rate': supply_false_positive_rate
        })
        
        print(f"  Overall Period Anomaly Rate: {overall_period_anomaly_rate:.2f}%")
        print(f"  Supply Recall (Non-Regular supplies identified): {supply_recall_non_regular:.3f}")
        print(f"  Supply False Positive Rate (Regular supplies flagged): {supply_false_positive_rate:.3f}")
        print(f"  Raw Counts: TP={true_positive_supplies_count}, FP={false_positive_supplies_count}, FN={false_negative_supplies_count}, TN={true_negative_supplies_count}")


        # Optimization metric: Maximize supply recall while penalizing supply false positives
        # Using the weighted approach: 0.7 for recall, 0.3 for false positives
        # Define weights (Hyperparameter tune according to case (prioritize identifying known frauds or allowing more false positives))
        a = 0.7
        b = 0.3
        current_score = (a * supply_recall_non_regular) - (b * supply_false_positive_rate)
        if current_score > best_score:
            best_score = current_score
            best_params = {'nu': nu_val, 'gamma': gamma_val}

results_df_eval = pd.DataFrame(evaluation_results)
print("\n--- Hyperparameter Evaluation Results (Supply-Level) ---")
# Displaying all columns to show raw counts as well
print(results_df_eval.sort_values(by='supply_recall_non_regular', ascending=False).to_string()) # .to_string() for full view

print(f"\nBest parameters found (based on weighted score): {best_params}")

# --- Optional: Visualize Tuning Results (using supply-level metrics) ---
plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
sns.lineplot(data=results_df_eval, x='nu', y='supply_recall_non_regular', hue='gamma', marker='o')
plt.title('Supply Recall (Non-Regular Supplies) vs. Nu')
plt.xlabel('nu (Expected Anomaly Proportion)')
plt.ylabel('Supply Recall on Non-Regular Supplies')
plt.grid(True)
plt.xscale('log') # Use log scale for nu if values span orders of magnitude

plt.subplot(1, 2, 2)
sns.lineplot(data=results_df_eval, x='nu', y='supply_false_positive_rate', hue='gamma', marker='o')
plt.title('Supply False Positive Rate (Regular Supplies) vs. Nu')
plt.xlabel('nu (Expected Anomaly Proportion)')
plt.ylabel('Supply False Positive Rate')
plt.grid(True)
plt.xscale('log') # Use log scale for nu
plt.tight_layout()
plt.show()


### 4. Final One-Class SVM Model Training and Application

'Once the best hyperparameters are identified, the final model is trained and applied to the entire dataset to generate anomaly predictions and scores.'

print(f"\n--- Training Final OCSVM Model with nu={best_params['nu']}, gamma={best_params['gamma']} ---")

final_oc_svm_model = OneClassSVM(kernel='rbf', nu=best_params['nu'], gamma=best_params['gamma'])
final_oc_svm_model.fit(X_scaled)

# Add predictions and scores to the merged DataFrame
merged_df['anomaly_prediction_ocsvm'] = final_oc_svm_model.predict(X_scaled)
merged_df['anomaly_score_ocsvm'] = final_oc_svm_model.decision_function(X_scaled)

print("\n--- Final Merged DataFrame Head with OCSVM Results ---")
print(merged_df[['Supply_ID', 'Period_Start_Date', 'Period_End_Date',
                 'Mean_Consumption_Period', 'Is_Non_Regular',
                 'anomaly_prediction_ocsvm', 'anomaly_score_ocsvm']].head())

print("\n--- Counts of OCSVM Anomaly Predictions (Period-Level) ---")
print(merged_df['anomaly_prediction_ocsvm'].value_counts())


### 5. Detailed Supply-Level Cross-Checking and Examples

'This section provides a detailed breakdown of the models performance at the supply level using the confusion matrix components, which are most relevant to your projects goal.'

print("\n--- Detailed Supply-Level Cross-Checking for Best Model ---")

# Identify all supplies that OCSVM flagged with at least one anomaly using the final model
ocsvm_flagged_supplies_final = merged_df[merged_df['anomaly_prediction_ocsvm'] == -1]['Supply_ID'].unique()

# True Positives (TP_Supplies): Non-Regular Supplies correctly identified
true_positive_supplies_final = np.intersect1d(ocsvm_flagged_supplies_final, all_non_regular_supply_ids)
print(f"Number of 'Non-Regular' Supplies correctly identified by OCSVM (TP): {len(true_positive_supplies_final)}")

# False Positives (FP_Supplies): Regular Supplies incorrectly flagged (potential new anomalies)
false_positive_supplies_final = np.intersect1d(ocsvm_flagged_supplies_final, all_regular_supply_ids)
print(f"Number of 'Regular' Supplies incorrectly flagged by OCSVM (FP - potential new anomalies): {len(false_positive_supplies_final)}")

# False Negatives (FN_Supplies): Non-Regular Supplies missed by OCSVM
false_negative_supplies_final = np.setdiff1d(all_non_regular_supply_ids, ocsvm_flagged_supplies_final)
print(f"Number of 'Non-Regular' Supplies missed by OCSVM (FN): {len(false_negative_supplies_final)}")

# True Negatives (TN_Supplies): Regular Supplies correctly not flagged
true_negative_supplies_final = np.setdiff1d(all_regular_supply_ids, ocsvm_flagged_supplies_final)
print(f"Number of 'Regular' Supplies correctly not flagged by OCSVM (TN): {len(true_negative_supplies_final)}")

# --- Display examples for investigation ---
print("\n--- Examples of True Positive Supplies (OCSVM flagged and are Non-Regular) ---")
if len(true_positive_supplies_final) > 0:
    example_tp_supplies_df = merged_df[merged_df['Supply_ID'].isin(true_positive_supplies_final) & (merged_df['anomaly_prediction_ocsvm'] == -1)].sort_values(by='anomaly_score_ocsvm').head(10)
    print(example_tp_supplies_df[['Supply_ID', 'Period_Start_Date', 'Mean_Consumption_Period', 'Is_Non_Regular', 'anomaly_score_ocsvm', 'anomaly_prediction_ocsvm']])
else:
    print("No True Positive Supplies found.")

print("\n--- Examples of False Positive Supplies (OCSVM flagged but are Regular) ---")
if len(false_positive_supplies_final) > 0:
    example_fp_supplies_df = merged_df[merged_df['Supply_ID'].isin(false_positive_supplies_final) & (merged_df['anomaly_prediction_ocsvm'] == -1)].sort_values(by='anomaly_score_ocsvm').head(10)
    print(example_fp_supplies_df[['Supply_ID', 'Period_Start_Date', 'Mean_Consumption_Period', 'Is_Non_Regular', 'anomaly_score_ocsvm', 'anomaly_prediction_ocsvm']])
else:
    print("No False Positive Supplies found.")

print("\n--- Examples of False Negative Supplies (Non-Regular but OCSVM missed) ---")
if len(false_negative_supplies_final) > 0:
    example_fn_supplies_df = merged_df[merged_df['Supply_ID'].isin(false_negative_supplies_final)].head(10)
    print(example_fn_supplies_df[['Supply_ID', 'Period_Start_Date', 'Mean_Consumption_Period', 'Is_Non_Regular', 'anomaly_score_ocsvm', 'anomaly_prediction_ocsvm']])
else:
    print("No False Negative Supplies found.")


# ### 6. Saving the Model, Scaler, and Final Data

# 'Finally, the trained model, the scaler, and the updated DataFrame with anomaly results are saved for future use and analysis.'


# # Create directory for saving models if it doesn't exist
# os.makedirs(MODEL_DIR, exist_ok=True) 

# model_filename = os.path.join(MODEL_DIR, 'final_ocsvm_model.joblib')
# scaler_filename = os.path.join(MODEL_DIR, 'scaler.joblib')

# # Save the OCSVM model
# joblib.dump(final_oc_svm_model, model_filename)
# print(f"\nOCSVM model saved to: {model_filename}")

# # Save the StandardScaler
# joblib.dump(scaler, scaler_filename)
# print(f"Scaler saved to: {scaler_filename}")

# # Save the final merged_df with anomaly results
# merged_df.to_csv(OUTPUT_FINAL_CSV_PATH, index=False) # index=False to prevent 'Unnamed: 0' next time
# print(f'Final merged df with OCSVM results saved to: {OUTPUT_FINAL_CSV_PATH}')